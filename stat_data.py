import pandas as pd
import matplotlib.pyplot as plt

# Baca data dari CSV
file_csv_name = "prediction_results_ori.csv"
df = pd.read_csv(file_csv_name)

# 1. Hitung rata-rata waktu komputasi (time_ms)
average_time = df['time_ms'].mean()
print(f'Rata-rata waktu komputasi: {average_time:.2f} ms')

# 2. Tambahkan kolom untuk status benar/salah
df['is_correct'] = df['actual'] == df['prediksi']

# 3. Hitung jumlah tebakan benar dan salah
counts = df['is_correct'].value_counts()
print(f'Rekap Tebakan: {counts}')


