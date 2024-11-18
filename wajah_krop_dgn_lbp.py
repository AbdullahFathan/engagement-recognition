import time
import cv2
import torch
import numpy as np
from datetime import datetime
from skimage.feature import local_binary_pattern
from torchvision import transforms
from torch import nn
from PIL import Image
from facenet_pytorch import MTCNN
import os

# File text name
txt_name = "results_fathan_lbp_v2"

# Load the pre-trained model
model = torch.load('model/model_lbp_new.pth')
model.eval()

# Class labels
class_label = ["very low", "low", "high", "very high"]

# Directories for saving frames and logs
frame_path = f"frames/{txt_name}"
os.makedirs(frame_path, exist_ok=True)
os.makedirs("dokumentasi", exist_ok=True)

# Function to convert image to LBP
def lbp_transform(image):
    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Extract LBP features
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")

    # Normalize LBP to 0-255
    lbp_normalized = np.uint8(255 * (lbp / lbp.max()))

    # Convert back to tensor with 3 channels
    lbp_tensor = torch.from_numpy(lbp_normalized).unsqueeze(0).repeat(3, 1, 1)
    return lbp_tensor

# Define preprocessing transformations
data_transforms = transforms.Compose([
    transforms.Lambda(lbp_transform),  # Apply LBP transformation
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Frame ID counter
frame_counter = 1

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start time for computation
        start_time = time.time()

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame)

        # If faces are detected, crop and process each face
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)

                # Add margin to the bounding box
                margin = 60  # Adjust the margin as needed
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)

                # Crop the face from the frame
                cropped_face = frame[y1:y2, x1:x2]

                # Convert cropped face to PIL image
                cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))

                # Apply transformations (with LBP)
                input_tensor = data_transforms(cropped_face_pil).unsqueeze(0)

                # Make prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    predicted_label = class_label[predicted]

                # Save frame to local storage with incremented ID
                frame_id = f"s-{frame_counter}"
                frame_filename = f"{frame_path}/{frame_id}.jpg"
                cv2.imwrite(frame_filename, frame)

                # End time for computation
                end_time = time.time()
                computation_time = end_time - start_time

                # Log the result with the frame ID instead of the activity timer
                result = f"Frame ID: {frame_id}  | Predicted: {predicted_label} | Computation Time: {computation_time:.4f} seconds"
                print(result)

                # Save result to a text file
                with open(f'dokumentasi/{txt_name}.txt', "a") as file:
                    file.write(result + "\n")

                # Increment the frame counter for the next frame
                frame_counter += 1

        else:
            print("No faces detected.")

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    # Release the camera and clean up
    cap.release()
    cv2.destroyAllWindows()
