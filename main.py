import os, torch, json
from process_img import Pipeline

def get_tensor(pipeline):
    data_dir = "facebank"
    tensor_dir = "facebank_tensor"
    
    facebank = {}
    
    for name in os.listdir(data_dir):
        name_bank = []
        for img_path in os.listdir(os.path.join(data_dir, name)):
            img = os.path.join(data_dir, name, img_path)

            embeddings_tensor = pipeline.process_arcface(img).numpy().tolist()
            name_bank.append(embeddings_tensor)
            # torch.save(embeddings_tensor, os.path.join(tensor_dir, name, f"{img_path[:-4]}_tensor.pt"))
        
        facebank[name] = name_bank
    
    with open("sample.json", "w") as outfile: 
        json.dump(facebank, outfile)

if __name__ == "__main__":
    pipeline = Pipeline(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # pipeline.process_img("yolo-sample-image.jpg")
    pipeline.process_camera()
    # get_tensor(pipeline=pipeline)