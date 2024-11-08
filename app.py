import os
import json
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO

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
    transforms.Resize((128, 128)),  # Resize to match model's input size
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
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

    # Ensure directories exist
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(os.path.join(upload_folder, 'annotated'), exist_ok=True)

    file.save(filepath)  # Save the uploaded file

    # Run the prediction based on the selected model
    if model_choice == 'yolo':
        result = yolov11_predict(yolo_model, filepath)
    elif model_choice == 'cnn':
        image = Image.open(filepath)
        image_tensor = transform(image).unsqueeze(0).to(device)
        result = cnn_predict(cnn_model, image_tensor)
    elif model_choice == 'faster_rcnn':
        image = Image.open(filepath)
        image_tensor = transform(image).unsqueeze(0).to(device)
        result = faster_rcnn_predict(faster_rcnn_model, image_tensor)
    else:
        return jsonify({'error': 'Invalid model selected'}), 400

    # Convert result to a Python list if it's a JSON string
    if isinstance(result, str):
        result = json.loads(result)

    # Draw bounding boxes on the image and get the annotated image path
    annotated_image_path = draw_bounding_boxes(filepath, result)
    annotated_image_url = url_for('static', filename=f'uploads/annotated/{filename}')

    # Log paths for debugging
    print(f"Original image path: {filepath}")
    print(f"Annotated image path: {annotated_image_path}")
    print(f"Annotated image URL: {annotated_image_url}")

    # Include the annotated image URL and result in the response
    response = {'result': result, 'image_url': annotated_image_url}
    
    return jsonify(response)


# YOLOv11 Prediction Function
def yolov11_predict(model, image_path, conf_threshold=0.5, iou_threshold=0.4):
    results = model.predict(image_path, conf=conf_threshold, iou=iou_threshold)
    predictions = results[0]  # Get predictions for the first (and only) image
    boxes = predictions.boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = predictions.boxes.conf.cpu().numpy()  # Confidence scores
    labels = predictions.boxes.cls.cpu().numpy()  # Class labels

    formatted_results = []
    for box, score, label in zip(boxes, scores, labels):
        formatted_results.append({
            'box': box.tolist(),     # Bounding box as [x_min, y_min, x_max, y_max]
            'label': int(label),     # Class label
            'score': float(score)    # Confidence score
        })

    return formatted_results

# CNN Prediction (for classification)
def cnn_predict(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
    return {'class': predicted.item(), 'confidence': confidence}

# Faster R-CNN Prediction (for object detection)
def faster_rcnn_predict(model, image):
    model.eval()
    with torch.no_grad():
        image = image.to("cpu")
        model = model.to("cpu")
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

def draw_bounding_boxes(image_path, predictions):
    """
    Draw bounding boxes on the image and save it in the 'annotated' folder.
    
    Args:
        image_path (str): Path to the original image.
        predictions (list): List of predictions, where each prediction is a dictionary
                            with keys "box" (bounding box coordinates), "label", and "score".
    
    Returns:
        str: Path to the annotated image.
    """
    # Open the image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Optional: Load a font for drawing text (comment out if not available)
    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except IOError:
        font = ImageFont.load_default()

    # Iterate over each prediction and draw the bounding box
    for pred in predictions:
        box = pred["box"]
        score = pred["score"]
        label = pred["label"]
        
        # Draw the box and text only for high-confidence predictions
        if score >= 0.5:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            text = f"Label {label}: {score:.2f}"
            text_size = draw.textsize(text, font=font)
            draw.rectangle([x_min, y_min - text_size[1], x_min + text_size[0], y_min], fill="red")
            draw.text((x_min, y_min - text_size[1]), text, fill="white", font=font)

    # Save the annotated image in the 'annotated' folder within 'uploads'
    annotated_image_path = image_path.replace("uploads", "uploads/annotated")
    os.makedirs(os.path.dirname(annotated_image_path), exist_ok=True)
    image.save(annotated_image_path)
    print(annotated_image_path)

    return annotated_image_path

if __name__ == '__main__':
    app.run(debug=True)
