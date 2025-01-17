import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained MobileNet model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)  # Adjust number of classes accordingly
model.load_state_dict(torch.load('mobilenet_crack_detect.pth', map_location=device))
model = model.to(device)
model.eval()

# Define transforms for preprocessing
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to draw bounding box
def draw_bounding_box(frame, label, confidence, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide the video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and preprocess
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = data_transforms(rgb_frame).unsqueeze(0).to(device)

    # Predict using the model
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][preds].item()
    
    # Assuming class 0 is "No Crack" and class 1 is "Crack"
    label_map = {0: "No Crack", 1: "Crack"}
    label = label_map[preds.item()]
    
    # Draw bounding box with label and confidence
    height, width, _ = frame.shape
    x, y, w, h = int(0.1 * width), int(0.1 * height), int(0.8 * width), int(0.8 * height)
    draw_bounding_box(frame, label, confidence, x, y, w, h)

    # Show frame
    cv2.imshow('Detection', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
