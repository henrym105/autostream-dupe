# from inference import get_model
from inference.models.utils import get_model
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

model = get_model(model_id="yolov8n-640", api_key=api_key)

results = model.infer("https://media.roboflow.com/inference/people-walking.jpg")

print(type(results))
print(results)