import json, torch
import torchvision.transforms as T
from utils.tools import *
from mtcnn.tools import *
from model import *
from sklearn.metrics.pairwise import cosine_similarity
from config import REFERENCE_POINT

class Pipeline : 
    def __init__(self, device='cpu', facebank_path='facebank.json'):
        self.mtcnn = initialize_mtcnn(device=device)
        self.arcface = initialize_arcface(device=device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(facebank_path, "r") as outfile: 
            self.facebank = json.load(outfile)
    
    def process_face(self, img, box, point):
        cropped_img = mtcnn_crop(img, box)
        normalized_point = normalize_point(point, box)
        
        aligned_face = align_face(cropped_img, normalized_point, REFERENCE_POINT)
        input_tensor = img2tensor(aligned_face)
        
        with torch.no_grad():
            embeddings = self.arcface(input_tensor)
        
        return embeddings

    def recognize_face(self, embedding):
        best_match = None
        best_score = -1

        for name, known_embedding in self.facebank.items():
            # Calculate cosine similarity
            similarity = cosine_similarity(embedding.cpu().numpy(), np.array(known_embedding).reshape(1, -1))[0][0]

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
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return annotated_img

    # def process_img2(self, img) : 
    #     results = mtcnn_detect(img, self.mtcnn)

    #     if results['boxes'] is not None and results['points'] is not None:
    #         annotated_img = img.copy()

    #         for box, point in zip(results['boxes'], results['points']):
    #             # Get embedding for detected face
    #             embedding = self.process_face(img, box, point)
    #             identity = self.recognize_face(embedding)

    #             # Draw annotation
    #             x1, y1, x2, y2 = map(int, box)
    #             cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             cv2.putText(annotated_img, f"{identity}", (x1, y1 - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    #         return annotated_img
    
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
            
            try : 
                annotated_frame = self.process_img(frame)
            except : 
                annotated_frame = frame

            cv2.imshow("Live Face Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()