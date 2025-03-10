import torch
import os
from flask import Flask, request, jsonify
from PIL import Image
from model import model_loader  # Import model loader
from werkzeug.utils import secure_filename

# Load BLIP Model
blip_model, blip_processor = model_loader.get_models()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Flask App
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    
    # Process Image
    image = Image.open(filepath).convert("RGB")
    caption = generate_caption(image)
    os.remove(filepath)  # Clean up uploaded file

    return jsonify({"caption": caption})

def generate_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    app.run(debug=True)
