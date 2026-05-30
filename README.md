# Tomato Leaf Disease Detection Web App

Aplikasi demo berbasis Streamlit untuk mendeteksi penyakit daun tomat menggunakan model **EfficientNetV2B0** dan visualisasi **Grad-CAM**.

## Cara Instalasi

1. Pastikan Python 3.11+ sudah terinstall.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Cara Menjalankan

Jalankan perintah berikut di terminal:
```bash
streamlit run app.py
```

## Fitur
- Deteksi 4 kelas (Early Blight, Late Blight, Leaf Mold, Healthy).
- Visualisasi Grad-CAM untuk transparansi keputusan model.
- Tabel probabilitas untuk analisis detail.
- Filter confidence threshold (< 60%).
