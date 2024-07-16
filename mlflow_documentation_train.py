import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

def main():
    # Create or set an MLflow experiment
    experiment_name = "RandomForestExperiment"
    mlflow.set_experiment(experiment_name)
    
    # Load dataset
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=42)
    
    # Start an MLflow run
    run = mlflow.start_run()
    
    try:
        # Log parameters
        n_estimators = 100
        mlflow.log_param("n_estimators", n_estimators)
        
        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        
        # Make predictions and log metrics
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric("mse", mse)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run ID: {run.info.run_id}")
    finally:
        # End the run
        mlflow.end_run()

if __name__ == "__main__":
    main()
