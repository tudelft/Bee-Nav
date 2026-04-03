import numpy as np
import onnxruntime as ort
import time
from PIL import Image
import torchvision.transforms as transforms

def predict_onnx(img_path, onnx_model_path):
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    image = Image.open(img_path)
    transform = transforms.Compose([
        transforms.Resize((192, 1800)),
        transforms.ToTensor()
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)
    
    input_data = image.numpy()

    input_name = session.get_inputs()[0].name
    
    time_start = time.time()
    
    outputs = session.run(None, {input_name: input_data})
    
    time_end = time.time()
    print(f"Inference time (ONNX): {time_end - time_start:.4f} seconds")

    output_vec = outputs[0][0] 

    prediction = np.arctan2(output_vec[1], output_vec[0])
    distance = np.linalg.norm(output_vec)

    return prediction, distance

if __name__ == "__main__":
    onnx_path = "networks/gazenet_mydata.onnx" 
    img_path = "test_image.jpg"
    
    pred, dist = predict_onnx(img_path, onnx_path)
    print(f"Heading: {pred}, Distance: {dist}")
