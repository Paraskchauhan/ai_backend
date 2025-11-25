from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import torch

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = torch.load("best.pt", map_location="cpu")
model.eval()

@app.route("/")
def home():
    return {"message": "Backend is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_base64 = request.json["image"]
        image_base64 = image_base64.split(",")[1]

        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes))

        results = model(img)

        if len(results.xyxy[0]) > 0:
            cls = int(results.xyxy[0][0][-1])
            label = results.names[cls]
        else:
            label = "no_object"

        return jsonify({"result": label})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
