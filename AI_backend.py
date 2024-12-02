'''AI Backend Service
The AI backend is responsible for processing the image input, performing object detection using a lightweight model, and returning the results in a structured format.

Responsibilities:
Accept image input from the UI backend.
Run the image through an object detection model (YOLO, MobileNet SSD, etc.).
Return detection results (bounding boxes, labels, confidence scores) in JSON format.
Tech Stack:
Python (Flask/FastAPI): For serving the API and interacting with the object detection model.
OpenCV/TensorFlow/PyTorch: Libraries for object detection.
YOLO or MobileNet SSD: Lightweight pre-trained object detection models.
Implementation Outline (Flask Example): '''
import cv2
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load pre-trained lightweight detection model (e.g., MobileNet SSD)
model = tf.saved_model.load("saved_model_path")

def detect_objects(image):
    # Preprocess the image for model input
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run object detection
    detections = model(input_tensor)
    
    # Process the output (bounding boxes, labels, confidence scores)
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy()
    
    return boxes, scores, classes

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file part"}), 400
    
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Run object detection
    boxes, scores, classes = detect_objects(img)
    
    # Return results in JSON format
    results = []
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Only return detections with confidence > 0.5
            result = {
                'class': str(classes[i]),
                'score': float(scores[i]),
                'box': boxes[i].tolist()
            }
            results.append(result)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

