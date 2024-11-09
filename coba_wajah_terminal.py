import time

# Start time for the activity
overall_start_time = time.time()


import cv2
import torch
import numpy as np
from datetime import datetime
from skimage.feature import local_binary_pattern
from torchvision import transforms
from torch import nn
from PIL import Image

# File text name
txt_name = "classification_results_fathan"

# Load the pre-trained model
model = torch.load('model/model_lbp.pth')
model.eval()

# Class labels
class_label = ["very low", "low", "high", "very high"]

# Define preprocessing transformations
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to convert image to LBP version
def convert_to_lbp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    lbp = np.uint8(255 * lbp / np.max(lbp))
    return lbp

# Initialize camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start time for computation
        start_time = time.time()
        
        # Convert frame to LBP format
        lbp_image = convert_to_lbp(frame)
        
        # Convert LBP image to PIL format
        lbp_pil_image = Image.fromarray(lbp_image)
        
        # Apply transformations
        input_tensor = data_transforms(lbp_pil_image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = class_label[predicted]
        
        # End time for computation
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Print the result
        activity_timer = time.time() - overall_start_time
        result = f"Activity Timer: {activity_timer:.4f} seconds | Predicted: {predicted_label} | Computation Time: {computation_time:.4f} seconds"
        print(result)
        
        # Save result to a text file
        with open(f'dokumentasi/{txt_name}.txt', "a") as file:
            file.write(result + "\n")

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    # Release the camera and clean up
    cap.release()
    cv2.destroyAllWindows()
