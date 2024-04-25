import torch
import numpy as np
from cnn import *

def CNN_make_model():
    # fine-tuning
    model_fine_tuning = ResNet18(pretrained=True, probing=False)
    train_baseline(model_fine_tuning, 1, 0.0001, 32)
    torch.save(model_fine_tuning.state_dict(), 'model.pth')
    model_predict(model_fine_tuning)
    
def run_model():
    model = ResNet18(pretrained=True, probing=False)
    model.load_state_dict(torch.load('model.pth'))
    model_predict(model)

def train_ready_model():
    model = ResNet18(pretrained=False, probing=False)
    model.load_state_dict(torch.load('model.pth'))
    model_predict(model)
    train_baseline(model, 10, 0.0001, 32)
    model_predict(model)
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    CNN_make_model()
    #run_model()
    #train_ready_model()