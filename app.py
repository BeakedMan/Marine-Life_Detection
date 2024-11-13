import os
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

# Class names for various models
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

class_names_no_background = {
    0: "Fish",
    1: "ball",
    2: "circle cage",
    3: "cube",
    4: "cylinder",
    5: "human body",
    6: "metal bucket",
    7: "plane",
    8: "rov",
    9: "square cage",
    10: "tyre",
}

class_names_no_background_yolo = {
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

# For Ensemble Model (9 classes)
class_names_ensemble_9 = {
    0: "Ball",
    1: "Circle Cage",
    2: "Cube",
    3: "Cylinder",
    4: "Fish",
    5: "Human Body",
    6: "Metal Bucket",
    7: "Square Cage",
    8: "Tyre",
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

# Define the EnsembleModel class
class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        # Initialize ResNet50 and modify the final layer
        self.modelA = models.resnet50(pretrained=False)
        self.modelA.fc = nn.Linear(self.modelA.fc.in_features, num_classes)
        
        # Initialize VGG16 and modify the final layer
        self.modelB = models.vgg16(pretrained=False)
        self.modelB.classifier[6] = nn.Linear(self.modelB.classifier[6].in_features, num_classes)

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        return (x1 + x2) / 2  # Average the predictions

# Load YOLO model
yolo_model = YOLO('models/best.pt')

# Initialize and load the ensemble model
num_classes = len(class_names_ensemble_9)
ensemble_model = EnsembleModel(num_classes=num_classes)
ensemble_model.load_state_dict(torch.load('models/ensemble_model.pth', map_location=device))
ensemble_model.to(device)

# Load the custom CNN model
cnn_model = SimpleCNN(num_classes=11)
cnn_model.load_state_dict(torch.load('models/cnn_model.pth', map_location=device))
cnn_model.to(device)

# Load Faster R-CNN model and adjust the classifier head
faster_rcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features
faster_rcnn_model.roi_heads.box_predictor.cls_score = nn.Linear(in_features, len(class_names_all))
faster_rcnn_model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features, len(class_names_all) * 4)
faster_rcnn_model.load_state_dict(torch.load('models/faster_rcnn_model.pth', map_location=device))
faster_rcnn_model.to(device)

# Define transformations for input images
transform = transforms.Compose([
    transforms.ToTensor(),
])

transform_ensemble = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

cnn_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model')
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'annotated'), exist_ok=True)
    file.save(filepath)

    if model_choice == 'yolo':
        result = yolov11_predict(yolo_model, filepath)
        annotated_image_path = draw_bounding_boxes(filepath, result, class_names_no_background_yolo)
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
        response = {'result': result}

    elif model_choice == 'ensemble':
        image = Image.open(filepath)
        image_tensor = transform_ensemble(image).unsqueeze(0).to(device)
        result = ensemble_predict(ensemble_model, image_tensor)
        response = {'result': result}

    else:
        return jsonify({'error': 'Invalid model selected'}), 400

    return jsonify(response)

def yolov11_predict(model, image_path, conf_threshold=0.3, iou_threshold=0.4):
    results = model.predict(image_path, conf=conf_threshold, iou=iou_threshold)
    predictions = results[0]
    boxes = predictions.boxes.xyxy.cpu().numpy()
    scores = predictions.boxes.conf.cpu().numpy()
    labels = predictions.boxes.cls.cpu().numpy()

    formatted_results = []
    for box, score, label in zip(boxes, scores, labels):
        formatted_results.append({
            'box': box.tolist(),
            'label': int(label + 1),
            'score': float(score)
        })
    return formatted_results

def cnn_predict(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_name = class_names_no_background.get(predicted.item(), f"Class {predicted.item()}")
        return {'class': class_name, 'confidence': confidence.item()}

def faster_rcnn_predict(model, image):
    model.eval()
    with torch.no_grad():
        predictions = model(image)
    return format_predictions(predictions)

def ensemble_predict(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_name = class_names_ensemble_9.get(predicted.item(), f"Class {predicted.item()}")
        return {'class': class_name, 'confidence': confidence.item()}

def format_predictions(predictions):
    results = []
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        results.append({
            'box': box.tolist(),
            'label': label.item(),
            'score': score.item()
        })
    return results

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
    image.save(annotated_image_path)
    
    return annotated_image_path

if __name__ == '__main__':
    app.run(debug=True)
