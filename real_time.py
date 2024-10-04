import cv2
import torch
import torchvision.transforms as transforms
from resnet import ResNet34
from utils import load_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model = ResNet34(num_classes=10).to(device)  # Modify num_classes according to your dataset
model = load_model(model, 'models/resnet34.pth')

# Define the transformation for input image (same as training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# OpenCV Video Capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

def predict_frame(frame, model):
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    input_tensor = transform(frame_rgb).unsqueeze(0).to(device)  # Add batch dimension
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.item()

# Define class labels (CIFAR-10 example)
class_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

while True:
    ret, frame = cap.read()  # Read a frame from the video feed
    
    if not ret:
        break
    
    # Predict the class for the current frame
    predicted_class = predict_frame(frame, model)
    label = class_labels[predicted_class]
    
    # Display the prediction on the frame
    cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame
    cv2.imshow('Real-Time ResNet34 Inference', frame)
    
    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
