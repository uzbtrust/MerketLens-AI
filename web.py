import os
# Segmentation fault xatosini oldini olish
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1' 

import faiss
import torch
import streamlit as st
import time
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# 1. Sahifa sozlamalari
st.set_page_config(page_title="MarketLens AI", layout="wide", initial_sidebar_state="collapsed")

# 2. CSS - Toza oq va minimalist dizayn
st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    .main {background-color: #ffffff;}
    
    /* Metrika yozuvlarini neytral rangda qilish */
    [data-testid="stMetricValue"] > div { color: #ffffff !important; font-weight: 600; }
    [data-testid="stMetricLabel"] > div { color: #666666 !important; }
    
    /* Sarlavhalar */
    h1, h2, h3 { color: #ffffff !important; font-family: 'Inter', sans-serif; }
    
    /* Rasm ostidagi matn */
    .stCaption { text-align: center; color: #444444; }
    </style>
    """, unsafe_allow_html=True)

# 3. Resurslarni yuklash
@st.cache_resource
def load_resources():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    weights = ResNet50_Weights.DEFAULT
    base_model = resnet50(weights=weights)
    model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
    model = model.to(device).eval()
    
    index = faiss.read_index("gallery.index")
    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
    return model, index, names, device

model, index, names, device = load_resources()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Asosiy Interfeys
st.title("🔍 MarketLens AI")
st.markdown("###### Mahsulotlarni qidirish va tahlil qilish tizimi")

# Filtrlash paneli
f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
with f_col1:
    uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
with f_col2:
    conf_level = st.slider("Moslik darajasi (%)", 0.0, 1.0, 0.4, step=0.05)
with f_col3:
    max_k = st.slider("Ko'rsatish limiti", 1, 24, 12)

st.markdown("---")

# 5. Qidiruv jarayoni
if uploaded_file:
    start_time = time.time()
    query_img = Image.open(uploaded_file).convert('RGB')
    
    # Vektorga o'girish
    img_t = transform(query_img).unsqueeze(0).to(device)
    with torch.no_grad():
        query_vec = model(img_t).cpu().view(1, -1).numpy().astype('float32')
        faiss.normalize_L2(query_vec)
    
    # Qidiruv
    distances, indices = index.search(query_vec, 100)
    search_time_ms = (time.time() - start_time) * 1000

    # Tizim statistikasi
    st.write("### 📊 Qidiruv statistikasi")
    s1, s2, s3, s4 = st.columns([1, 2, 2, 2])
    with s1:
        st.image(query_img, width=100)
    with s2:
        st.metric("Obyektlar soni", f"{len(names):,} ta")
    with s3:
        st.metric("Topish tezligi", f"{search_time_ms:.2f} ms")
    with s4:
        st.metric("Tanlangan aniqlik", f"{conf_level*100:.0f}% +")

    st.markdown("---")

    # Galereya
    st.write("##### 📦 Topilgan natijalar:")
    grid = st.columns(6)
    found_count = 0
    
    for i, idx in enumerate(indices[0]):
        similarity = 1 - (distances[0][i] / 2)
        if similarity >= conf_level and found_count < max_k:
            img_name = names[idx]
            img_path = os.path.join('data/gallery', img_name)
            with grid[found_count % 6]:
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), use_container_width=True)
                    st.caption(f"Moslik: **{similarity*100:.1f}%**")
                    found_count += 1
    
    if found_count == 0:
        st.warning("Berilgan parametrlar bo'yicha hech narsa topilmadi.")
else:
    st.info("💡 Davom etish uchun mahsulot rasmini yuklang.")