import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from MedViT import MedViT_small, MedViT_base, MedViT_large

def main(model_name):
    
    NUM_EPOCHS = 30
    BATCH_SIZE = 30

    # Determine the model to use
    if model_name == 'small':
        model = MedViT_small()
        checkpoint_path = './checkpoints/MedViT_small_im1k.pth'
    elif model_name == 'base':
        model = MedViT_base()
        checkpoint_path = './checkpoints/MedViT_base_im1k.pth'
    elif model_name == 'large':
        model = MedViT_large()
        checkpoint_path = './checkpoints/MedViT_large_im1k.pth'
    else:
        raise ValueError("Invalid model name. Choose 'small', 'base', or 'large'.")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=2, bias=True)
    model = model.cuda()

    # Set up data loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root='DDI_HAM10000/train', transform=transform)
    val_dataset = ImageFolder(root='DDI_HAM10000/val', transform=transform)
    test_dataset = ImageFolder(root='DDI_HAM10000/test', transform=transform)

    data_loader_train = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_val = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    data_loader_test = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup the training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)


    # Train the model
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(data_loader_train, desc=f"Epoch {epoch + 1}"):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader_val:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(data_loader_train)}, Validation Loss: {val_loss / len(data_loader_val)}, Accuracy: {100 * correct / total}%')

    # save the model
    torch.save(model.state_dict(), f"ham10000_{model_name}.pth")
    
    # Testing
    test_model(model, data_loader_test, model_name)

def test_model(model, data_loader, model_name):
    model.eval()
    total = 0
    correct = 0
    all_targets = []
    all_outputs = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_targets, all_outputs)
    f1 = f1_score(all_targets, all_outputs)

        # Calculate metrics
    accuracy = accuracy_score(all_targets, all_outputs)
    f1 = f1_score(all_targets, all_outputs, average='binary')
    auc = roc_auc_score(all_targets, all_probs)
    report = classification_report(all_targets, all_outputs, target_names=['Class 0', 'Class 1'])

    # Append results to the file
    with open("results.txt", "a") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"F1: {f1}\n")
        file.write(f"AUC: {auc}\n")
        file.write(f"Classification Report:\n{report}\n\n")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py [model_name]")
    else:
        main(sys.argv[1])
