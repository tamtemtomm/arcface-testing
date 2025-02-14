import cv2
import numpy as np
from PIL import Image, ImageDraw
from utils.detect_faces import detect_face, extract_face

def mtcnn_detect(img:np.ndarray, model):
  boxes, probs, points = model.detect(img, landmarks=True)
  return {
      "boxes" : boxes,
      "probs" : probs,
      "points": points
    }  

def mtcnn_annotates(img: np.ndarray, boxes:np.ndarray, points:np.ndarray, width:int=10, annotate=False, save_path = None):
  img = Image.fromarray(img)
  img_draw = img.copy()

  draw = ImageDraw.Draw(img_draw)
  for i, (box, point) in enumerate(zip(boxes, points)):
        draw.rectangle(box.tolist(), width=5)
        if annotate : 
          for p in point:
              draw.rectangle((p - width).tolist() + (p + width).tolist(), width=width)
        extract_face(img, box)

  if save_path is not None :
    img_draw.save(save_path)

  return img_draw

def mtcnn_annotate(img: np.ndarray, box:np.ndarray, point:np.ndarray, width:int=10, annotate=False, save_path = None):
  img = Image.fromarray(img)
  img_draw = img.copy()

  draw = ImageDraw.Draw(img_draw)
  draw.rectangle(box.tolist(), width=5)
  if annotate : 
    for p in point:
        draw.rectangle((p - width).tolist() + (p + width).tolist(), width=width)
  # extract_face(img, box)

  # Crop the face from the bounding box
  x1, y1, x2, y2 = map(int, box)
  img_cropped = img.crop((x1, y1, x2, y2))  # Crop only the face
  
  if save_path is not None :
    img_cropped.save(save_path)

  img_cropped = np.array(img_cropped)
  return img_cropped

def align_face(face, face_points, reference_points):
  H, W, _ = np.float32(face).shape
  src_points = np.float32(face_points)

  # Use fixed reference points (no need to normalize by W/H)
  reference_points_fixed = np.float32(reference_points)

  # Calculate the Affine Transform matrix
  M = cv2.estimateAffinePartial2D(src_points, reference_points_fixed, method=cv2.LMEDS)[0]

  # Apply the transformation (make sure output size is correct)
  aligned_face = cv2.warpAffine(face, M, (W, H), flags=cv2.INTER_LINEAR)

  img = Image.fromarray(aligned_face)
  img.save("aligned_face.jpg")

  return aligned_face