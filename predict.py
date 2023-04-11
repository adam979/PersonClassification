from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import json

image_path = 'C:/Users/hassa/Desktop/PersonClassification/Image Processing/ARRahman/Image 2.jpg'

model = models.resnet18(pretrained=False)
num_classes = 4
model.fc = nn.Linear(512, num_classes)

# Load the model state from the JSON file
with open("model.json", "r") as f:
    state_dict_json = json.load(f)

# Convert the state_dict_json values back to PyTorch tensors
state_dict = {k: torch.tensor(v) for k, v in state_dict_json.items()}

# Load the state_dict into the model
model.load_state_dict(state_dict)
model.eval()

# Define the data transforms used during training
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load the image and apply data transforms
image = Image.open(image_path)
image = image.convert('RGB')
image = data_transforms(image)
image = image.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).float()  # Convert image to float and move to the appropriate device

# Move the model to the appropriate device (CPU or GPU)
model = model.to(image.device)

# Make a prediction
with torch.no_grad():
    output = model(image.unsqueeze(0))
    _, predicted = output.max(1)

dataset = ImageFolder('C:/Users/hassa/Desktop/PersonClassification/Image Processing/', transform=data_transforms)
class_names = dataset.classes
predicted_class_name = class_names[predicted.item()]

# Print the predicted label
print('Predicted label:', predicted.item())
# Print the predicted class name
print('Predicted class name:', predicted_class_name)


image_np = image.cpu().numpy().transpose((1, 2, 0))

# Display the image with predicted label as title
plt.imshow(image_np)
plt.title('Predicted Class: {}'.format(predicted_class_name))
plt.axis('off')
plt.show()


