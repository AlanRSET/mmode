import streamlit as st
from pathlib import Path
import keras
from PIL import Image, ImageOps
import numpy as np
import h5py

st.title("M mode image classification - VGG16 Model")
st.header("")
st.text("Upload a Image for image classification")

from img_classification import teachable_machine_classification

import urllib.request

url = 'https://github.com/AlanRSET/mmode/releases/download/vgg1/VGG16Mmodegood.h5'
hf = url.split('/')[-1]

urllib.request.urlretrieve(url, hf)


        
uploaded_file = st.file_uploader("Choose an Mmode image ...", type=['png','jpeg','jpg'])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        
        #hf=load_model()
        #hf = h5py.File('F:/streamlit-trial/VGG16Mmodegood.h5', 'r')
        label = teachable_machine_classification(image, hf)
        if label == 0:
            st.write("The scan is of congestion")#cardeogenic pulmonary edema
        elif label == 1:
            st.write("The scan is of normal lung")
        else:
            st.write("The scan is of pleural effusion")
        #st.write("The scan is of1 ", label)
