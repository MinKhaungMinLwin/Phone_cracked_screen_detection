import os
import torch
import torch.nn as nn
import torch.optim as  optim
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')

data_dir = 'Data' # Train foler

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

image_paths = [sample[0] for sample in full_dataset.samples]
labels = [sample[1] for sample in full_dataset.samples]

# Split into train and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.05, stratify=labels, random_state=42
)

# Recreate datasets
train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

from collections import Counter
class_counts = Counter([sample[1] for sample in full_dataset.samples])
print(class_counts)

train_dataset.samples = list(zip(train_paths, train_labels))
val_dataset.samples = list(zip(val_paths, val_labels))

train_size = int(0.95 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.mobilenet_v2(pretrained=True)

model.classifier[1] = nn.Linear(model.last_channel, len(full_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

early_stoppin_patience = 10
best_val_loss = float('inf')
early_stoppin_counter = 0

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    global  best_val_loss, early_stoppin_counter
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"train loss: {train_loss:.3f} Acc : {train_acc:.3f}")

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        print(f"Val loss : {val_loss:.3f} Acc : {val_acc:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stoppin_counter = 0
            torch.save(model.state_dict(), 'mobilenet_crack_detect.pth')
        else:
            early_stoppin_counter += 1
            if early_stoppin_counter >= early_stoppin_patience:
                print("Early stopping triggered.")
                break

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100)

def evaluate_model(model, val_loader):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / len(val_loader.dataset)
    print(f"Validation Accuracy : {accuracy:.3f}")

model.load_state_dict(torch.load('mobilenet_crack_detect.pth'))
evaluate_model(model, val_loader)
