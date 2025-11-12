from flask import Flask, request, jsonify
from utils.model_utils import load_model, save_model
from utils.data_utils import append_data, load_data
from utils.trainer import train_model, get_metrics
import os

app = Flask(__name__)

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load model (create if not found)
model = load_model()

@app.route('/')
def home():
    return jsonify({"NeoMind": "Online", "status": "ready", "version": "v1"})

@app.route('/upload-data', methods=['POST'])
def upload_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    append_data(data)
    return jsonify({"status": "Data added", "entries": len(data)})

@app.route('/train', methods=['POST'])
def train():
    global model
    dataset = load_data()
    if not dataset:
        return jsonify({"error": "No training data"}), 400
    logs = train_model(model, dataset)
    save_model(model)
    return jsonify({"status": "Training complete", "logs": logs[-3:]})

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(get_metrics())

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"model_loaded": True, "training_ready": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
