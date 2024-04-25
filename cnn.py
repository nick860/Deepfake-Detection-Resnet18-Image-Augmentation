import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch 
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained: # if we want to use the pretrained model
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transfrom = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights) 
        else:
            self.transfrom = transforms.Compose([transforms.Resize((244, 244)), transforms.ToTensor()])
            self.resnet18 = resnet18() # this is the resnet model

        in_features_dim = self.resnet18.fc.in_features # get the input features of the model
        self.resnet18.fc = nn.Identity() # remove the last layer of the model
        if probing:
            for name, param in self.resnet18.named_parameters():
                param.requires_grad = False

        self.logistic_regression = nn.Linear(in_features_dim, 1) # add a logistic regression layer to the model

        
    def forward(self, x):
        features = self.resnet18(x) 
        return self.logistic_regression(features)
    
def compute_accuracy(model, data_loader, device):
    model.eval()
    ### YOUR CODE HERE ###
    correct = 0
    total = 0
    prediction = []
    with torch.no_grad():
        for (imgs, labels) in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs).squeeze()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            prediction.append((torch.sigmoid(outputs) > 0.5).int())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
         
    return correct / total , torch.cat(prediction, 0).cpu().numpy()

def get_data_loaders(transform, path, batch_size):
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def run_training_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device) # img in shape (batch_size, 3, 224, 224), labels in shape (batch_size)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze() # squeeze from shape (batch_size, 1) to (batch_size)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

def model_predict(model):
      transform = model.transfrom
      path = os.path.join(os.getcwd(), 'whichfaceisreal') # get the path of the current directory
      batch_size = 32
      train_loader, val_loader, test_loader = get_data_loaders(transform, path, batch_size) # get the data loaders
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # get the device
      model = model.to(device) # move the model to the device
      test_acc, prediction = compute_accuracy(model, test_loader, device)
      print(f'Accuracy: {test_acc}')

def train_baseline(model, num_epochs, learning_rate, batch_size):
    test_acc = 0
    transform = model.transfrom
    path = os.path.join(os.getcwd(), 'whichfaceisreal') # get the path of the current directory
    train_loader, val_loader, test_loader = get_data_loaders(transform, path, batch_size) # get the data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # get the device
    model = model.to(device) # move the model to the device
    criterion = torch.nn.BCEWithLogitsLoss() # define the loss function, that is Binary Cross Entropy with Logits
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001) # define the optimizer
    for _ in range(num_epochs):
        run_training_epoch(model, criterion, optimizer, train_loader, device)
        test_acc, prediction = compute_accuracy(model, test_loader, device)
        print(f'Test accuracy: {test_acc:.4f}')
   