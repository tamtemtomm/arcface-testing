import os, torch
from process_img import Pipeline

def get_tensor(pipeline):
    data_dir = "facebank"
    tensor_dir = "facebank_tensor"
    for name in os.listdir(data_dir):
        for img_path in os.listdir(os.path.join(data_dir, name)):
            img = os.path.join(data_dir, name, img_path)
            print(img)
            embeddings_tensor = pipeline.process_arcface(img)
            torch.save(embeddings_tensor, os.path.join(tensor_dir, name, f"{img_path[:-4]}_tensor.pt"))

if __name__ == "__main__":
    pipeline = Pipeline(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    pipeline.process("yolo-sample-image.jpg")