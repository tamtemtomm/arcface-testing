import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

def img2np(img:str):
  img_np = plt.imread(img)
  
  if img.endswith(".png"):
    img_np = np.uint8(img_np * 255)
  
  return img_np

def img2tensor(img:np.ndarray, size=(112, 112), mean=0.5, std=0.5): 
  img = Image.fromarray(img)

  transform = T.Compose([
      T.Resize(size),  # Resize to 112x112 (ArcFace standard)
      T.ToTensor(),           # Convert to tensor
      T.Normalize(mean=[mean, mean, mean], std=[std, std, std])  # Normalize to [-1, 1]
  ])

    # Apply the transforms
  tensor = transform(img).unsqueeze(0)  # Add batch dimension
  return tensor

def display(img:np.ndarray):
  plt.imshow(img)

def save_image(img : np.ndarray, filepath:str):
  # Save img using PIL
  pil_img = Image.fromarray(img)
  pil_img.save(filepath)

def normalize_point(point, box):
  x1, y1, x2, y2 = map(int, box)
  
  normalized_point = point.copy()
  for i, p in enumerate(point) : 
    normalized_point[i][0] = (p[0] - x1)/(x2 - x1) * 100
    normalized_point[i][1] = (p[1] - y1)/(y2 - y1) * 100
  
  return normalized_point

def detect_faces_yolo(img:str, model):
  img = img2np(img) 
  results = model(img)  # Run face detection

  faces = []
  for i, result in enumerate(results):
      for box in result.boxes.xyxy:  # Bounding box format [x1, y1, x2, y2]
          x1, y1, x2, y2 = map(int, box)
          face = img[y1:y2, x1:x2]  # Crop face
          faces.append(face)

  return faces