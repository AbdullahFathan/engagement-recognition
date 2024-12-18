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
from facenet_pytorch import MTCNN
import os

# File text name
txt_name = "ori_percobaan5"

# Load the pre-trained model
model = torch.load('model/model_ori_p2.pth')
model.eval()

# Class labels
class_label = ["very low", "low", "high", "very high"]

frame_path = f"frames/{txt_name}"
os.makedirs(frame_path, exist_ok=True)



# Define preprocessing transformations
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
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

                # Convert to PIL image
                cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))

                # Apply transformations
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

                activity_timer = time.time() - overall_start_time

                # Log the result with the frame ID instead of the activity timer
                result = f"Activity Time: {activity_timer} | Frame ID: {frame_id}  | Predicted: {predicted_label} | Computation Time: {computation_time:.4f} seconds"
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
