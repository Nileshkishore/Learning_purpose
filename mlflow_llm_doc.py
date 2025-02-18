import tiktoken
import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import mlflow
from dotenv import load_dotenv
import time
from datetime import datetime
import gradio as gr
import threading

# Load .env file
load_dotenv()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Ensure collection is created with Cosine Similarity
try:
    collection = chroma_client.get_collection(name="sports_articles")
except chromadb.exceptions.CollectionNotFoundError:
    collection = chroma_client.create_collection(name="sports_articles", metadata={"hnsw:space": "cosine"})

# Azure OpenAI Configuration
openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)

# Set up MLflow experiment
mlflow.set_experiment("sports_articles_openai_integration")

def calculate_detailed_costs(system_prompt, user_query, response_text):
    """
    Calculate detailed costs for input and output tokens for Azure OpenAI API usage.
    """
    # Initialize tiktoken encoder
    encoding = tiktoken.encoding_for_model("gpt-4o")
    
    # Calculate input tokens
    system_tokens = len(encoding.encode(system_prompt))
    query_tokens = len(encoding.encode(user_query))
    total_input_tokens = system_tokens + query_tokens
    
    # Calculate output tokens
    output_tokens = len(encoding.encode(response_text))
    
    # Cost rates per 1000 tokens
    INPUT_RATE = 0.0005  # $0.0005 per 1K tokens
    OUTPUT_RATE = 0.0015  # $0.0015 per 1K tokens
    
    # Calculate costs
    input_cost = (total_input_tokens / 1000) * INPUT_RATE
    output_cost = (output_tokens / 1000) * OUTPUT_RATE
    total_cost = input_cost + output_cost
    
    return {
        "token_breakdown": {
            "system_prompt_tokens": system_tokens,
            "user_query_tokens": query_tokens,
            "total_input_tokens": total_input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_input_tokens + output_tokens
        },
        "cost_breakdown": {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6)
        },
        "rates": {
            "input_rate_per_1k": INPUT_RATE,
            "output_rate_per_1k": OUTPUT_RATE
        }
    }

def retrieve_documents(query_text):
    """Retrieve relevant documents using Cosine Similarity"""
    try:
        query_embedding = model.encode(query_text).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        documents = results["documents"] if results["documents"] else []
        metadata = results["metadatas"] if results["metadatas"] else []
        return documents, metadata
    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return [], []

def generate_answer(query_text):
    """Generate an answer using Azure OpenAI GPT-4o-mini"""
    documents, metadata = retrieve_documents(query_text)

    if not documents:
        return "Sorry, I couldn't find relevant information in the database.", [], [], "", 0, ""

    # Prepare context
    context_text = "\n\n".join([str(doc)[:500] for doc in documents[:3]])

    # Prepare system prompt
    system_prompt = f"""You are an intelligent assistant that provides 
    answers based on retrieved sports articles. Use the following context 
    to answer the user's query.\n\nContext:\n{context_text}"""

    try:
        start_time = time.time()
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_text},
            ],
            max_tokens=300,
            temperature=0.7
        )
        latency = time.time() - start_time
        response_text = response.choices[0].message.content
        return response_text, documents, metadata, system_prompt, latency, response_text
    except Exception as e:
        print(f"Error during OpenAI API call: {str(e)}")
        return "An error occurred while generating the answer.", [], [], "", 0, ""

def get_prompt_version(query_text):
    """Generate a unique version for each prompt"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"prompt_v{timestamp}"

def log_to_mlflow_with_costs(prompt_version, query_text, answer, documents, metadata, system_prompt, latency, response_text):
    """Log the details with comprehensive cost analysis to MLflow"""
    with mlflow.start_run(run_name=f"run_{prompt_version}") as run:
        # Calculate detailed costs
        cost_analysis = calculate_detailed_costs(system_prompt, query_text, response_text)
        
        # Log basic parameters
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_param("user_prompt", query_text)
        mlflow.log_param("model_used", "gpt-4o-mini")
        
        # Log token metrics
        mlflow.log_metrics({
            "system_prompt_tokens": cost_analysis["token_breakdown"]["system_prompt_tokens"],
            "user_query_tokens": cost_analysis["token_breakdown"]["user_query_tokens"],
            "total_input_tokens": cost_analysis["token_breakdown"]["total_input_tokens"],
            "output_tokens": cost_analysis["token_breakdown"]["output_tokens"],
            "total_tokens": cost_analysis["token_breakdown"]["total_tokens"]
        })
        
        # Log cost metrics
        mlflow.log_metrics({
            "input_cost": cost_analysis["cost_breakdown"]["input_cost"],
            "output_cost": cost_analysis["cost_breakdown"]["output_cost"],
            "total_cost": cost_analysis["cost_breakdown"]["total_cost"],
            "latency_seconds": round(latency, 3),
            "response_length": len(response_text)
        })
        
        # Log other metrics
        mlflow.log_metrics({
            "max_tokens": 300,
            "temperature": 0.7
        })
        
        # Log system prompt and response
        mlflow.log_param("system_prompt", system_prompt)
        mlflow.log_param("response_text", response_text)
        
        # Log documents info
        mlflow.log_param("documents_found", len(documents))
        
        # Save and log response
        response_path = "run_artifacts/response.txt"
        os.makedirs("run_artifacts", exist_ok=True)
        with open(response_path, "w") as f:
            f.write(response_text)
        mlflow.log_artifact(response_path, "outputs")
        
        # Log source code
        current_file = os.path.abspath(__file__)
        mlflow.log_artifact(current_file, "source_code")

def gradio_interface(query_text):
    start_time = time.time()
    
    prompt_version = get_prompt_version(query_text)
    answer, documents, metadata, system_prompt, latency, response_text = generate_answer(query_text)
    
    before_mlflow = time.time()
    print(f"Time before MLflow: {before_mlflow - start_time} seconds")
    
    # MLflow logging
    threading.Thread(
        target=log_to_mlflow_with_costs,
        args=(prompt_version, query_text, answer, documents, metadata, 
              system_prompt, latency, response_text)
    ).start()
    
    after_mlflow = time.time()
    print(f"Time to start MLflow thread: {after_mlflow - before_mlflow} seconds")
    
    return answer

## [Previous imports and functions remain the same until create_gradio_interface()]

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Sports Articles Q&A System")
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Enter your question about sports:",
                    placeholder="e.g., Who won the latest Super Bowl?",
                    lines=3
                )
                submit_btn = gr.Button("Get Answer")
            
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=5,
                    interactive=False  # Use interactive=False instead of readonly
                )
        
        # Handle submission
        submit_btn.click(
            fn=gradio_interface,
            inputs=query_input,
            outputs=answer_output
        )
    
    return demo

# Launch application
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
