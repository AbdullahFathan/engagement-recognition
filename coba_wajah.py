import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from skimage.feature import local_binary_pattern
from torchvision import transforms
from torch import nn

# Memuat kembali seluruh model
model = torch.load('model_complete.pth',)
model.eval()

class_label = ["very low", "low", "high", "very high" ]

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

# Function to capture frame from camera and process it
def capture_frame():
    ret, frame = cap.read()
    if ret:
        # Tampilkan frame asli dari kamera di GUI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ubah dari BGR (OpenCV) ke RGB (PIL)
        frame_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        lbp_image_label.config(image=frame_tk)
        lbp_image_label.image = frame_tk

        # Proses deteksi dengan LBP (tetapi tidak ditampilkan)
        lbp_image = convert_to_lbp(frame)
        
        # Konversi LBP ke format PIL untuk deteksi model
        lbp_pil_image = Image.fromarray(lbp_image)

        # ini kalau mau langsung nyoba dari input gambar ke model
        # lbp_pil_image = Image.fromarray(frame_rgb)
        
        # Terapkan transformasi sebelum deteksi
        input_tensor = data_transforms(lbp_pil_image).unsqueeze(0)
        
        # Lakukan prediksi model
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            print(f"Ini hasil dari {class_label[predicted]} ")
            label = "Predicted Label: " + class_label[predicted]
        
        # Update label prediksi di GUI
        prediction_label.config(text=label)
    
    # Lakukan capture frame lagi setelah 10ms
    root.after(10, capture_frame)

# Initialize GUI
root = tk.Tk()
root.title("Camera Input with LBP and Prediction")

# Initialize camera
cap = cv2.VideoCapture(1)

# Set the frame rate to 30 FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# Create a label to display the original image
lbp_image_label = Label(root)
lbp_image_label.pack()

# Create a label to display the prediction result
prediction_label = Label(root, text="Predicted Label: ")
prediction_label.pack()

# Start capturing frames
capture_frame()

# Run the Tkinter main loop
root.mainloop()

# Release the camera when the window is closed
cap.release()
cv2.destroyAllWindows()
