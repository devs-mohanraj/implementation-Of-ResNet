import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from resnet import ResNet34

model_path = "resnet34_cifar10.pth"
test_dir = 'test/images'
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def download_and_save_test_images():
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    test_dataset = CIFAR10(root='./data', train=False, download=True)
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        image_path = os.path.join(test_dir, f'{i}_{class_names[label]}.png')
        image.save(image_path)
    print(f"Downloaded and saved {len(test_dataset)} images to {test_dir}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet34(num_classes=10).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

def load_and_transform_image(image_path):
    image = Image.open(image_path)
    image_transformed = transform(image)  # Convert the image to a tensor
    image_transformed = image_transformed.unsqueeze(0)  # Add batch dimension
    return image, image_transformed.to(device)

def test_and_show_images():
    image_files = os.listdir(test_dir)
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            original_image, transformed_image = load_and_transform_image(image_path)
            
            with torch.no_grad():
                outputs = model(transformed_image)
                _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
                
                predicted_class = class_names[predicted.item()]  # Convert the tensor to a class name
            
            plt.imshow(original_image)
            plt.title(f'Predicted: {predicted_class}')
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    download_and_save_test_images()
    test_and_show_images()
