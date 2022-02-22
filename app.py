import streamlit as st
import numpy as np
from PIL import Image
from fastai.vision.all import load_learner, Path
from urllib.request import urlretrieve
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

url = ("http://dl.dropboxusercontent.com/s/sclyjuhgakarplh/export.pkl?raw=1")
filename = "export.pkl"
urlretrieve(url,filename)

urll = ("http://dl.dropboxusercontent.com/s/ecl4tj6q2u8s4q3/fig-03_5.png?raw=1")
filenamee = "1.png"
urlretrieve(urll,filenamee)
st.image(filenamee)

st.title("NUMBER CLASSIFIER")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
learn_inf = load_learner(Path("export.pkl"))
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Your Image.', use_column_width=True)
    image = np.asarray(img)
    label = learn_inf.predict(image) 

    st.write("")
    st.write("Classifying...")

    if label[0][0] in "AEIOU":
        st.write("## This looks like an")
    else:
        st.write("## This looks like a")

    st.title(label[0].lower().title())
