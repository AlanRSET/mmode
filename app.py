import streamlit as st
from pathlib import Path
import keras
from PIL import Image, ImageOps
import numpy as np
import h5py

st.title("M mode image classification")
st.header("")
st.text("Upload a Image for image classification")

from img_classification import teachable_machine_classification

cloud_model_location = "https://drive.google.com/file/d/1o9_Y96ZvfBnPUT5p-wPvdJ_8kq-Gq-wK/view?usp=sharing"

@st.cache
def load_model():

    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("model/VGG16Mmodegood.h5")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)
    
    model = torch.load(f_checkpoint, map_location=device)
    model.eval()
    return model


uploaded_file = st.file_uploader("Choose an Mmode image ...", type=['png','jpeg','jpg','dcm'])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        hf=load_model()
        #hf = h5py.File('F:/streamlit-trial/VGG16Mmodegood.h5', 'r')
        label = teachable_machine_classification(image, hf)
        if label == 0:
            st.write("The scan is of congestion")#cardeogenic pulmonary edema
        elif label == 1:
            st.write("The scan is of normal lung")
        else:
            st.write("The scan is of pleural effusion")
        #st.write("The scan is of1 ", label)
