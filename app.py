import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
import sys

# Tambahkan folder src ke path agar bisa import modul lokal
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from predict import preprocess_image, get_prediction, CLASS_NAMES
    from gradcam import make_gradcam_heatmap, save_and_display_gradcam
except ImportError as e:
    st.error(f"Gagal mengimport modul pendukung: {e}")

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Komparasi Penyakit Tomat",
    page_icon="🍅",
    layout="wide"
)

# Inisialisasi session state untuk menyimpan hasil analisis agar bisa digabung
if 'eff_results' not in st.session_state:
    st.session_state.eff_results = None
if 'mob_results' not in st.session_state:
    st.session_state.mob_results = None

# --- LOAD MODELS ---
@st.cache_resource
def load_model_eff():
    path = 'outputs/model/best_efficientnet.keras'
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

@st.cache_resource
def load_model_mob():
    path = 'outputs/model/best_mobilenet.keras'
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>Klasifikasi Penyakit Daun Tomat</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Perbandingan EfficientNetV2 dan MobileNetV3 Menggunakan Grad-CAM</h4>", unsafe_allow_html=True)
st.divider()

model_eff = load_model_eff()
model_mob = load_model_mob()

# --- DUA KOLOM ---
col1, col2 = st.columns(2)

# --- KOLOM KIRI (EFFICIENTNETV2) ---
with col1:
    st.markdown("### EfficientNetV2")
    if model_eff is None:
        st.error("Model EfficientNetV2 tidak ditemukan.")
    else:
        uploaded_eff = st.file_uploader("Pilih gambar daun tomat (EffV2)", type=["jpg", "jpeg", "png"], key="eff_upload")

        if uploaded_eff:
            image_eff = Image.open(uploaded_eff)
            st.image(image_eff, caption="Preview Citra Input", use_container_width=True)

            if st.button("🚀 ANALISIS EFFICIENTNET", key="btn_eff"):
                with st.spinner("Menganalisis..."):
                    img_array = preprocess_image(image_eff)
                    label, conf, probs = get_prediction(model_eff, img_array)

                    # Simpan ke session state
                    st.session_state.eff_results = {
                        'label': label,
                        'conf': conf,
                        'probs': probs
                    }

                    # Tampilkan Hasil
                    st.success(f"**Prediksi:** {label}")
                    st.write(f"**Confidence Score:** {conf*100:.2f}%")
                    st.progress(float(conf))

                    # Grad-CAM
                    try:
                        base_model = model_eff.layers[0]
                        last_conv = [l.name for l in base_model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
                        heatmap = make_gradcam_heatmap(img_array, model_eff, last_conv)

                        temp_in = "temp_eff_in.jpg"
                        res_out = "res_eff_gradcam.jpg"
                        image_eff.save(temp_in)
                        save_and_display_gradcam(temp_in, heatmap, res_out)

                        st.markdown("**Visualisasi Grad-CAM:**")
                        st.image(res_out, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Grad-CAM tidak tersedia: {e}")

                    # Probabilitas
                    st.markdown("**Probabilitas Seluruh Kelas (EffV2):**")
                    prob_df = pd.DataFrame(list(probs.items()), columns=['Penyakit', 'Skor'])
                    prob_df = prob_df.sort_values(by='Skor', ascending=False)
                    st.table(prob_df.style.format({'Skor': '{:.4f}'}))

# --- KOLOM KANAN (MOBILENETV3) ---
with col2:
    st.markdown("### MobileNetV3")
    if model_mob is None:
        st.error("Model MobileNetV3 tidak ditemukan.")
    else:
        uploaded_mob = st.file_uploader("Pilih gambar daun tomat (MobV3)", type=["jpg", "jpeg", "png"], key="mob_upload")

        if uploaded_mob:
            image_mob = Image.open(uploaded_mob)
            st.image(image_mob, caption="Preview Citra Input", use_container_width=True)

            if st.button("🚀 ANALISIS MOBILENET", key="btn_mob"):
                with st.spinner("Menganalisis..."):
                    img_array = preprocess_image(image_mob)
                    label, conf, probs = get_prediction(model_mob, img_array)

                    # Simpan ke session state
                    st.session_state.mob_results = {
                        'label': label,
                        'conf': conf,
                        'probs': probs
                    }

                    # Tampilkan Hasil
                    st.info(f"**Prediksi:** {label}")
                    st.write(f"**Confidence Score:** {conf*100:.2f}%")
                    st.progress(float(conf))

                    # Grad-CAM
                    try:
                        base_model = model_mob.layers[0]
                        last_conv = [l.name for l in base_model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
                        heatmap = make_gradcam_heatmap(img_array, model_mob, last_conv)

                        temp_in = "temp_mob_in.jpg"
                        res_out = "res_mob_gradcam.jpg"
                        image_mob.save(temp_in)
                        save_and_display_gradcam(temp_in, heatmap, res_out)

                        st.markdown("**Visualisasi Grad-CAM:**")
                        st.image(res_out, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Grad-CAM tidak tersedia: {e}")

                    # Probabilitas
                    st.markdown("**Probabilitas Seluruh Kelas (MobV3):**")
                    prob_df = pd.DataFrame(list(probs.items()), columns=['Penyakit', 'Skor'])
                    prob_df = prob_df.sort_values(by='Skor', ascending=False)
                    st.table(prob_df.style.format({'Skor': '{:.4f}'}))

# --- KOMPARASI ANALISIS REAL-TIME ---
if st.session_state.eff_results and st.session_state.mob_results:
    st.divider()
    st.markdown("## Komparasi Analisis Real-Time")
    st.info("Tabel di bawah menggabungkan skor probabilitas dari kedua model untuk gambar yang baru saja dianalisis.")

    # Ambil data dari session state
    eff_p = st.session_state.eff_results['probs']
    mob_p = st.session_state.mob_results['probs']

    # Buat DataFrame Gabungan
    comparison_data = []
    for name in CLASS_NAMES:
        comparison_data.append({
            'Penyakit': name,
            'EfficientNetV2': eff_p.get(name, 0),
            'MobileNetV3': mob_p.get(name, 0)
        })

    df_compare = pd.DataFrame(comparison_data)

    # Tambahkan baris Keputusan Akhir
    st.markdown("### Perbandingan Skor Probabilitas")
    st.table(df_compare.style.format({
        'EfficientNetV2': '{:.4f}',
        'MobileNetV3': '{:.4f}'
    }))

    # Ringkasan Keputusan
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Prediksi EffV2", st.session_state.eff_results['label'],
                  f"{st.session_state.eff_results['conf']*100:.2f}%")
    with c2:
        st.metric("Prediksi MobV3", st.session_state.mob_results['label'],
                  f"{st.session_state.mob_results['conf']*100:.2f}%")

# --- PERBANDINGAN HASIL (DARI CSV) ---
st.divider()
st.markdown("## Perbandingan Metrik Evaluasi (Dataset Testing)")

comp_path = 'outputs/comparison/model_comparison_complete.csv'
if os.path.exists(comp_path):
    df_comp = pd.read_csv(comp_path)

    # Preprocessing nama model untuk tampilan
    df_comp['Model'] = df_comp['Model'].replace({
        'efficientnet': 'EfficientNetV2',
        'mobilenet': 'MobileNetV3'
    })

    # Ambil kolom yang relevan
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Size_MB', 'Avg_Inference_ms']
    df_display = df_comp[cols].copy()

    # Format metrik
    for m in ['Accuracy', 'Precision', 'Recall', 'F1-score']:
        df_display[m] = df_display[m].apply(lambda x: f"{x*100:.2f}%")

    # Rename kolom agar lebih rapi
    df_display.columns = ['Arsitektur', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Model Size (MB)', 'Inf. Time (ms)']

    st.table(df_display)
else:
    st.warning("File perbandingan tidak ditemukan.")

st.markdown("<br><hr><center><small>Penelitian Skripsi 2024 - Klasifikasi Penyakit Daun Tomat</small></center>", unsafe_allow_html=True)
