import torch
from arcface.model import Backbone
from mtcnn.model import MTCNN

def initialize_mtcnn(device='cpu', **kwargs) : 
    model = MTCNN(device=device, **kwargs) 
    return model 

def initialize_arcface(device="cpu") : 
    model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
    # Load the weights
    weight_path = 'model_weights/model_ir_se50.pth'  # Update this with your actual path
    model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)))

    model.eval()  # Set the model to evaluation mode
    
    return model