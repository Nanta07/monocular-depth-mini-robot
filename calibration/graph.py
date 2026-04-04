import numpy as np
import matplotlib.pyplot as plt

# Parameter Kalibrasi
a = 281.39
b = 0.16
R2 = 0.6494

# Membuat data kurva
predictions_ai = np.linspace(50, 600, 100)
calculated_distance = a * (1 / (predictions_ai + 1e-6)) + b

plt.figure(figsize=(10, 6))
plt.plot(predictions_ai, calculated_distance, label=f'Model Reciprocal ($R^2$ = {R2:.4f})', color='blue', linewidth=2.5)

# --- BAGIAN PERUBAHAN: RED DOT ---
# Mencari titik di kurva yang paling dekat dengan jarak 1 meter
idx_kritis = (np.abs(calculated_distance - 1.0)).argmin()

# Plot Titik Merah
plt.plot(predictions_ai[idx_kritis], calculated_distance[idx_kritis], 'ro', markersize=10)

# Teks Keterangan (Posisinya sedikit di atas titik merah)
plt.text(predictions_ai[idx_kritis], calculated_distance[idx_kritis] + 0.4, 
         'ZONA KRITIS (1-3m)\nSENSITIVITAS TINGGI', 
         color='red', fontsize=10, fontweight='bold', ha='center')
# ---------------------------------

# Formatting Grafik
plt.title('Validasi Jarak: Prediksi MiDaS vs Jarak Nyata', fontsize=14, fontweight='bold')
plt.xlabel('Prediksi Mentah AI (Semakin Kanan = Semakin Dekat)', fontsize=12)
plt.ylabel('Jarak Hasil Konversi (Meter)', fontsize=12)
plt.axhline(y=1, color='green', linestyle=':', alpha=0.5)
plt.axhline(y=3, color='orange', linestyle=':', alpha=0.5)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.ylim(0, 6) # Batas tampilan agar fokus pada jarak dekat
plt.legend()

plt.tight_layout()
plt.show()