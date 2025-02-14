import torch
import torchvision.transforms as T
from utils.tools import *
from mtcnn.tools import *
from model import *
from config import REFERENCE_POINT

class Pipeline : 
    def __init__(self, device='cpu'):
        self.mtcnn = initialize_mtcnn(device=device)
        self.arcface = initialize_arcface(device=device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def process_face(self, img, box, point):
        cropped_img = mtcnn_annotate(img, box, point, save_path="test_image.jpg")
        normalized_point = normalize_point(point, box)
        
        cropped_img = img2np("test_image.jpg")
        aligned_face = align_face(cropped_img, normalized_point, REFERENCE_POINT)
        
        input_tensor = img2tensor(aligned_face)
        print(input_tensor)
        
        with torch.no_grad():
            embeddings = self.arcface(input_tensor)
        
        return embeddings

    def process_arcface(self, img):
        img = img2np(img)
        print(img)
        input_tensor = img2tensor(img)
        print(input_tensor)
        
        with torch.no_grad():
            embeddings = self.arcface(input_tensor)
        return embeddings
    
    def process(self, img:str):
        img = img2np(img)
        results = mtcnn_detect(img, self.mtcnn)
        
        print(results)
        for box, point in zip(results['boxes'], results['points']):
            embeddings = self.process_face(img, box, point)
        
        print("Image Embeddings:", embeddings)