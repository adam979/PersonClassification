import json
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold 
from torch.utils.data import DataLoader

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

data_dir = 'C:/Users/hassa/Desktop/PersonClassification/Image Processing'
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

num_folds = 5
train_size = 13
test_size = 3
batch_size = 16

skf = StratifiedKFold(n_splits=num_folds-1, shuffle=True, random_state=42) #n_splits=4 since n_splits cannot be greater than the number of members in each class.
fold = 1

fold_losses = []
fold_accuracies = []

for train_index, test_index in skf.split(dataset, dataset.targets):
    print(f'Fold {fold}')

    
    train_dataset = data.Subset(dataset, train_index)
    test_dataset = data.Subset(dataset, test_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    model = models.resnet18(pretrained = True)

    num_epochs = 25
    num_classes = 4
    model.fc = nn.Linear(512,num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()*images.size(0)
        
        epoch_loss = running_loss/len(train_dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total +=labels.size(0)

            correct += predicted.eq(labels).sum().item()
        
        test_loss = running_loss/ len(test_dataset)
        test_accuracy = correct/total
    print('Test Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_accuracy))
    fold_losses.append(test_loss)
    fold_accuracies.append(test_accuracy)

    fold+=1


# Save the model's state dictionary to a JSON file

# Convert the tensors in the state_dict() to regular Python data types
# state_dict = model.state_dict()
# state_dict_json = {k: v.tolist() for k, v in state_dict.items()}

# # Save the state_dict() as JSON
# with open("model.json", "w") as f:
#     json.dump(state_dict_json, f)

avg_loss = sum(fold_losses)/num_folds
avg_accuracy = sum(fold_accuracies)/num_folds

print('Average Test Loss: {:.4f}, Average Accuracy: {:.4f}'.format(avg_loss, avg_accuracy))