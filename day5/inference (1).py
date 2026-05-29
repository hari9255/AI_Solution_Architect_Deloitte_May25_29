import joblib
import json
import pandas as pd


def model_fn(model_dir):
    
    model = joblib.load(f"{model_dir}/bike-sharing-model.pkl")
    
    return model


def input_fn(request_body, request_content_type):
    
    if request_content_type == "application/json":
        
        data = json.loads(request_body)
        
        df = pd.DataFrame([data])
        
        return df
    
    raise Exception("Unsupported content type")


def predict_fn(input_data, model):
    
    prediction = model.predict(input_data)
    
    return prediction


def output_fn(prediction, content_type):
    
    result = {
        "prediction": prediction.tolist()
    }
    
    return json.dumps(result)
