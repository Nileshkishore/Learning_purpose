import os
import chromadb
from sentence_transformers import SentenceTransformer
import time

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or load the collection
collection = chroma_client.get_or_create_collection(name="sports_articles")

# Folder containing text files
folder_path = "00-Sports-Articles"

# Track the overall time for the embedding task
embedding_start_time = time.time()

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if not content:
            print(f"⚠️ Skipping empty file: {filename}")
            continue  # Skip empty files

        # Track time for the embedding process
        embedding_process_start_time = time.time()

        # Create embedding
        embedding = model.encode(content).tolist()

        # Check if the ID exists before deleting
        existing_docs = collection.get(ids=[filename])
        if existing_docs and "ids" in existing_docs and existing_docs["ids"]:
            collection.delete(ids=[filename])
            print(f"🗑️ Deleted existing embedding for: {filename}")

        # Store in ChromaDB
        collection.add(
            ids=[filename],  # Unique identifier
            embeddings=[embedding],
            documents=[content],  # Ensure full document text is stored
            metadatas=[{"filename": filename}]
        )
        print(f"✅ Stored {filename}")

        # Measure the embedding time
        embedding_process_end_time = time.time()
        print(f"⏱️ Time taken for embedding {filename}: {embedding_process_end_time - embedding_process_start_time:.4f} seconds")

# Track total time for the embedding task
embedding_end_time = time.time()
print(f"✅ All embeddings saved to ChromaDB.")
print(f"⏱️ Total time taken for embedding tasks: {embedding_end_time - embedding_start_time:.4f} seconds")
