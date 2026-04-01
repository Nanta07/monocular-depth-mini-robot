import h5py
import numpy as np

# Path ke file yang baru kamu pindahkan
dataset_path = 'dataset/nyu_depth_v2/nyu_depth_v2_labeled.mat'

try:
    with h5py.File(dataset_path, 'r') as f:
        # Menampilkan variabel utama dalam dataset
        print("Kunci Utama dalam Dataset:", list(f.keys()))
        
        # Cek dimensi data
        # f['images'] biasanya (N, C, W, H)
        print("Shape Images:", f['images'].shape) 
        print("Shape Depths (Meter):", f['depths'].shape)
        
        print("\nDataset berhasil terdeteksi dan siap digunakan!")
except Exception as e:
    print(f"Gagal membaca file: {e}")