import cv2
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label
import os
import time

# File text name
txt_name = "ori_percobaan_gui"
frame_path = f"frames/{txt_name}"
os.makedirs(frame_path, exist_ok=True)

# Load the pre-trained model
model = torch.load('model/model_ori_p2.pth')
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

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# GUI setup
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition GUI with Live Camera")
        self.cap = cv2.VideoCapture(1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Frame for displaying video
        self.video_frame = Label(self.root)
        self.video_frame.pack()

        # Label for predictions
        self.prediction_label = Label(self.root, text="Predicted Label: ", font=("Arial", 14))
        self.prediction_label.pack()

        # Start the activity timer
        self.overall_start_time = time.time()
        self.frame_counter = 1

        # Start capturing frames
        self.running = True
        self.capture_frame()

    def capture_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Detect faces in the frame
        boxes, _ = mtcnn.detect(frame)

        # Process detected faces
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # Add margin to the bounding box
                margin = 60
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)

                # Crop the face
                cropped_face = frame[y1:y2, x1:x2]
                cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))

                # Preprocess and predict
                input_tensor = data_transforms(cropped_face_pil).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    predicted_label = class_label[predicted]

                # Save the frame with bounding box
                frame_id = f"s-{self.frame_counter}"
                frame_filename = f"{frame_path}/{frame_id}.jpg"
                cv2.imwrite(frame_filename, frame)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Log the result
                end_time = time.time()
                computation_time = end_time - self.overall_start_time
                result = f"Frame ID: {frame_id} | Predicted: {predicted_label} | Computation Time: {computation_time:.4f} seconds"
                print(result)
                with open(f'dokumentasi/{txt_name}.txt', "a") as file:
                    file.write(result + "\n")

                self.frame_counter += 1

        # Convert frame to ImageTk for GUI display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        # Update prediction label
        self.prediction_label.config(text=f"Predicted Label: {predicted_label}" if boxes is not None else "No face detected")

        # Schedule the next frame
        self.root.after(10, self.capture_frame)

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()
