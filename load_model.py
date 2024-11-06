import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_model(model_name, model_version="latest"):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    model_uri = f"models:/{model_name}/{model_version}"
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Model loading failed; {e}")
        return None
    return loaded_model

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_name = "iris_model"
model_version = "2"

# Load the model from the Model Registry
loaded_model = load_model(model_name=model_name, model_version=model_version)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

result[:4]

