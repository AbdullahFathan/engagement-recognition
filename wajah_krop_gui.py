import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from skimage.feature import local_binary_pattern
from torchvision import transforms
from facenet_pytorch import MTCNN
import os
import time

# File text name
txt_name = "results_fathan_lbp_v2"
frame_path = f"frames/{txt_name}"
os.makedirs(frame_path, exist_ok=True)
os.makedirs("dokumentasi", exist_ok=True)

# Load the pre-trained model
model = torch.load('model/model_lbp_new.pth')
model.eval()

# Class labels
class_label = ["very low", "low", "high", "very high"]

# Function to apply LBP
def lbp_transform(image):
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    lbp_normalized = np.uint8(255 * (lbp / lbp.max()))
    lbp_tensor = torch.from_numpy(lbp_normalized).unsqueeze(0).repeat(3, 1, 1)
    return lbp_tensor

# Preprocessing pipeline
data_transforms = transforms.Compose([
    transforms.Lambda(lbp_transform),
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

# Frame counter
frame_counter = 1

# Save and log results
def save_and_log_result(frame_id, predicted_label, computation_time):
    result = f"Frame ID: {frame_id} | Predicted: {predicted_label} | Computation Time: {computation_time:.4f} seconds"
    print(result)
    with open(f'dokumentasi/{txt_name}.txt', "a") as file:
        file.write(result + "\n")

# GUI-based frame processing
def process_frame():
    global frame_counter
    ret, frame = cap.read()
    if ret:
        start_time = time.time()

        # Detect faces
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                margin = 60
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)

                # Crop and process face
                cropped_face = frame[y1:y2, x1:x2]
                cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
                input_tensor = data_transforms(cropped_face_pil).unsqueeze(0)

                # Prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    predicted_label = class_label[predicted]

                # Save frame
                frame_id = f"s-{frame_counter}"
                frame_filename = f"{frame_path}/{frame_id}.jpg"
                cv2.imwrite(frame_filename, frame)

                # Log result
                computation_time = time.time() - start_time
                save_and_log_result(frame_id, predicted_label, computation_time)

                # Update GUI prediction label
                prediction_label.config(text=f"Predicted: {predicted_label}")

                frame_counter += 1
        else:
            prediction_label.config(text="No faces detected.")

        # Display frame in GUI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        video_label.config(image=frame_tk)
        video_label.image = frame_tk

    # Repeat after 10 ms
    root.after(10, process_frame)

# Initialize GUI
root = tk.Tk()
root.title("LBP-Based Face Detection and Prediction")

# Create video display label
video_label = Label(root)
video_label.pack()

# Create prediction display label
prediction_label = Label(root, text="Predicted: ", font=("Helvetica", 16))
prediction_label.pack()

# Start frame processing
process_frame()

# Run the Tkinter main loop
root.mainloop()

# Cleanup resources on exit
cap.release()
cv2.destroyAllWindows()
