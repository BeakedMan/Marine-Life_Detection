import os
import json
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO

# For Faster R-CNN
class_names_all = {
    0: "background",
    1: "Fish",
    2: "ball",
    3: "circle cage",
    4: "cube",
    5: "cylinder",
    6: "human body",
    7: "metal bucket",
    8: "plane",
    9: "rov",
    10: "square cage",
    11: "tyre",
}

# For CNN and YOLO (no background class)
class_names_no_background = {
    1: "Fish",
    2: "ball",
    3: "circle cage",
    4: "cube",
    5: "cylinder",
    6: "human body",
    7: "metal bucket",
    8: "plane",
    9: "rov",
    10: "square cage",
    11: "tyre",
}

# Set up device: MPS for Apple Silicon (M1), or fallback to CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Define a custom CNN model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Assuming input image size of 128x128
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load models
yolo_model = YOLO('models/best.pt')  # Load YOLOv11 model
num_classes = 11  # Set to the number of classes in your CNN model
cnn_model = SimpleCNN(num_classes=num_classes)
cnn_model.load_state_dict(torch.load('models/cnn_model.pth', map_location=device))
cnn_model.to(device)

# Set the number of classes for Faster R-CNN to match the saved model
faster_rcnn_classes = 12

# Load Faster R-CNN and update the classifier head for the correct number of classes
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features
faster_rcnn_model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, faster_rcnn_classes)
faster_rcnn_model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, faster_rcnn_classes * 4)
faster_rcnn_model.load_state_dict(torch.load('models/faster_rcnn_model.pth', map_location=device))
faster_rcnn_model.to(device)

# Define transformations for input images
transform = transforms.Compose([
    transforms.ToTensor(),  # No resizing for Faster R-CNN and YOLO
])

cnn_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize only for CNN
        transforms.ToTensor(),
    ])

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model')  # Get the selected model from form data
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename
    upload_folder = app.config['UPLOAD_FOLDER']
    filepath = os.path.join(upload_folder, filename)
    
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(os.path.join(upload_folder, 'annotated'), exist_ok=True)

    file.save(filepath)  # Save the uploaded file

    # Run the prediction based on the selected model
    if model_choice == 'yolo':
        result = yolov11_predict(yolo_model, filepath)
        annotated_image_path = draw_bounding_boxes(filepath, result, class_names_no_background)
        annotated_image_url = url_for('static', filename=f'uploads/annotated/{filename}')
        response = {'result': result, 'image_url': annotated_image_url}

    elif model_choice == 'faster_rcnn':
        image = Image.open(filepath)
        image_tensor = transform(image).unsqueeze(0).to(device)
        result = faster_rcnn_predict(faster_rcnn_model, image_tensor)
        annotated_image_path = draw_bounding_boxes(filepath, result, class_names_all)
        annotated_image_url = url_for('static', filename=f'uploads/annotated/{filename}')
        response = {'result': result, 'image_url': annotated_image_url}

    elif model_choice == 'cnn':
        image = Image.open(filepath)
        image_tensor = cnn_transform(image).unsqueeze(0).to(device)
        result = cnn_predict(cnn_model, image_tensor)
        # Only return class and confidence for CNN
        response = {'result': result}
        
    else:
        return jsonify({'error': 'Invalid model selected'}), 400

    print(f"Original image path: {filepath}")

    return jsonify(response)

# YOLOv11 Prediction Function
def yolov11_predict(model, image_path, conf_threshold=0.5, iou_threshold=0.4):
    results = model.predict(image_path, conf=conf_threshold, iou=iou_threshold)
    predictions = results[0]
    boxes = predictions.boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = predictions.boxes.conf.cpu().numpy()  # Confidence scores
    labels = predictions.boxes.cls.cpu().numpy()  # Class labels

    formatted_results = []
    for box, score, label in zip(boxes, scores, labels):
        formatted_results.append({
            'box': box.tolist(),
            'label': int(label + 1),  # Adjust for YOLO's class numbering
            'score': float(score)
        })

    return formatted_results

# CNN Prediction (for classification)
def cnn_predict(model, image):
    model.eval()
    with torch.no_grad():
        # Debug: Check the initial shape of `image`
        print(f"Initial image shape: {image.shape}")
        
        # Ensure the image tensor has a batch dimension of 1
        if image.dim() == 3:  
            image = image.unsqueeze(0)
        elif image.size(0) != 1:
            raise ValueError("Expected a single image, but got a batch of images.")
        
        # Debug: Confirm the shape after adjustment
        print(f"Image shape after ensuring batch dimension: {image.shape}")
        
        outputs = model(image)  # Expected output shape should be [1, num_classes]
        
        # Debug: Print the shape of outputs to confirm
        print(f"Output shape from the model: {outputs.shape}")
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the predicted class and confidence score
        _, predicted = torch.max(probabilities, 1)
        
        # Use the predicted class index to fetch the corresponding probability
        confidence = probabilities[0, predicted.item()].item()
        class_name = class_names_no_background.get(predicted.item(), f"Class {predicted.item()}")
        
    print(class_name, confidence)
    return {'class': class_name, 'confidence': confidence}


# Faster R-CNN Prediction (for object detection)
def faster_rcnn_predict(model, image):
    model.eval()
    with torch.no_grad():
        predictions = model(image)
    return format_predictions(predictions)

# Helper to format predictions for object detection
def format_predictions(predictions):
    results = []
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        results.append({
            'box': box.tolist(),
            'label': label.item(),
            'score': score.item()
        })
    return results

from PIL import Image, ImageDraw, ImageFont

def draw_bounding_boxes(image_path, predictions, class_names):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except IOError:
        font = ImageFont.load_default()

    for pred in predictions:
        label = pred["label"]
        score = pred.get("score", None)
        class_name = class_names.get(label, f"Class {label}")
        
        if "box" in pred:
            box = pred["box"]
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            text = f"{class_name}: {score:.2f}" if score is not None else class_name
            text_size = draw.textsize(text, font=font)
            draw.rectangle([x_min, y_min - text_size[1], x_min + text_size[0], y_min], fill="red")
            draw.text((x_min, y_min - text_size[1]), text, fill="white", font=font)

    annotated_image_path = image_path.replace("uploads", "uploads/annotated")
    os.makedirs(os.path.dirname(annotated_image_path), exist_ok=True)
    image.save(annotated_image_path)
    
    return annotated_image_path

if __name__ == '__main__':
    app.run(debug=True)
