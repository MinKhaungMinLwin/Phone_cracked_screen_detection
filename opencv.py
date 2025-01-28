import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
from ultralytics import YOLO

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained YOLOv5 model for object detection
yolo_model = YOLO('yolov5s.pt')  # Ensure the YOLO model is downloaded

# Load the trained MobileNet model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 1)  # Adjust number of classes

# Load MobileNet weights
try:
    model.load_state_dict(torch.load('C:/Users/htayn/HNO/mobilenet_crack_detect.pth', map_location=device))
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

model = model.to(device)
model.eval()

# Define transforms for MobileNet preprocessing
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to draw bounding box and label
def draw_bounding_box(frame, label, confidence, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide the video file path

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect phone screen using YOLO
    results = yolo_model(frame)
    phone_detected = False

    detections = results[0].boxes  # Access detected objects
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class ID
        if cls == 67:  # 67 is the YOLO class ID for "cell phone"
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            phone_detected = True

            # Crop the phone screen region
            screen_crop = frame[y:y+h, x:x+w]

            # Preprocess the screen region
            input_tensor = data_transforms(screen_crop).unsqueeze(0).to(device)

            # Predict using the MobileNet model
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                confidence = torch.softmax(outputs, dim=1)[0][preds].item()

            # Assuming class 0 is "No Crack" and class 1 is "Crack"
            label_map = {0: "No Crack", 1: "Crack"}
            label = label_map[preds.item()]

            # Draw bounding box with label
            draw_bounding_box(frame, label, confidence, x, y, w, h)

    if not phone_detected:
        cv2.putText(frame, "No phone detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Phone Screen Detection', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
