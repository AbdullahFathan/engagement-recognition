import os
import cv2
import torch
import pandas as pd
import time  # Untuk mengukur waktu komputasi
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Memuat model yang sudah disimpan
model = torch.load('./model/model_ori.pth')
model.eval()

class_label = ["very low", "low", "high", "very high"]
file_csv_name = "prediction_results_ori.csv"

# Definisi transformasi PyTorch
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Direktori gambar untuk pengujian
test_image_dirs = {
    'very low': './test_image/engagement_0',
    'low': './test_image/engagement_1',
    'high': './test_image/engagement_2',
    'very high': './test_image/engagement_3',
}

# Inisialisasi list untuk menyimpan hasil
results = []
all_actual = []  # Menyimpan semua label aktual
all_predicted = []  # Menyimpan semua label prediksi

# Proses setiap gambar di folder
for actual_label, dir_path in test_image_dirs.items():
    for image_file in os.listdir(dir_path):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dir_path, image_file)

            # Mulai waktu komputasi
            start_time = time.time()

            # Baca gambar
            image = Image.open(image_path)

            # Terapkan transformasi langsung pada gambar asli
            input_tensor = data_transforms(image).unsqueeze(0)

            # Lakukan prediksi
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = class_label[predicted.item()]

            # Hitung waktu komputasi (dalam ms)
            elapsed_time = (time.time() - start_time) * 1000  # Konversi detik ke ms

            # Simpan hasil actual, prediksi, dan waktu komputasi
            results.append({
                'actual': actual_label,
                'prediksi': predicted_label,
                'time_ms': round(elapsed_time, 2)  # Dibulatkan ke 2 desimal
            })

            # Simpan untuk evaluasi metrik
            all_actual.append(actual_label)
            all_predicted.append(predicted_label)

# Buat DataFrame dari hasil prediksi
df = pd.DataFrame(results)

# Hitung metrik evaluasi
accuracy = accuracy_score(all_actual, all_predicted)
precision = precision_score(all_actual, all_predicted, average='weighted')
recall = recall_score(all_actual, all_predicted, average='weighted')
f1 = f1_score(all_actual, all_predicted, average='weighted')

# Tambahkan metrik evaluasi ke DataFrame
df_metrics = pd.DataFrame([{
    'accuracy': round(accuracy, 2),
    'precision': round(precision, 2),
    'recall': round(recall, 2),
    'f1_score': round(f1, 2)
}])

# Gabungkan hasil prediksi dengan metrik evaluasi
df_combined = pd.concat([df, df_metrics], ignore_index=True)

# Simpan ke file CSV
df_combined.to_csv(f'./dokumentasi/{file_csv_name}', index=False)

print(f"Hasil prediksi dan evaluasi telah disimpan ke '{file_csv_name}'")
