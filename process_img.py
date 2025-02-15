import json, torch
from torch import nn
import torchvision.transforms as T
from utils.tools import *
from mtcnn.tools import *
from model import *
from config import REFERENCE_POINT

class Pipeline : 
    def __init__(self, device='cpu', facebank_path='facebank.json'):
        self.mtcnn = initialize_mtcnn(device=device)
        self.arcface = initialize_arcface(device=device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pairwise = nn.PairwiseDistance(p=2)
        
        with open(facebank_path, "r") as outfile: 
            self.facebank = json.load(outfile)
    
    def recognize_face(self, img, box, point):
        cropped_img = mtcnn_crop(img, box)
        normalized_point = normalize_point(point, box)
        
        aligned_face = align_face(cropped_img, normalized_point, REFERENCE_POINT)
        input_tensor = img2tensor(aligned_face)
        
        with torch.no_grad():
            embedding = self.arcface(input_tensor)
        
        name = self.estimate_face(embedding=embedding)
        return name

    def estimate_face(self, embedding):
        best_match = None
        best_score = -1

        for name, known_embedding in self.facebank.items():
            # Calculate cosine similarity
            similarity = self.pairwise(embedding, torch.tensor(known_embedding))[0][0]

            if similarity > best_score:
                best_score = similarity
                best_match = name

        # Apply a threshold for recognition
        return best_match if best_score >= 0.5 else "Unknown"
    
    def process_arcface(self, img):
        img = img2np(img)
        input_tensor = img2tensor(img)
        
        with torch.no_grad():
            embeddings = self.arcface(input_tensor)
        return embeddings
    
    def process_img(self, img):
        if type(img) == str : 
            img = img2np(img)
            
        results = mtcnn_detect(img, self.mtcnn)
                        
        annotated_img = mtcnn_annotate(img, results['boxes'], results['points'])
        
        for box, point in zip(results['boxes'], results['points']):
            
            identity = self.recognize_face(img, box, point)
            x1, y1, x2, y2 = map(int, box)
            cv2.putText(annotated_img, f"{identity}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return annotated_img
    
    def process_camera(self):
        cap = cv2.VideoCapture(0)  # Open webcam (0 for default camera)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()  # Read frame from camera
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            annotated_frame = self.process_img(frame)
            
            # try : 
            #     annotated_frame = self.process_img(frame)
            # except : 
            #     annotated_frame = frame

            cv2.imshow("Live Face Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()