import streamlit as st
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import cv2
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors


# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('featurevector (1).pkl', 'rb')))
filenames = pickle.load(open('filenames (2).pkl', 'rb'))


# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Set page title
st.title("Fashion Recommendation System")
if not os.path.exists("uploads"):
    os.mkdir("uploads")

# Function to save uploaded image
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# Function to extract features from image
def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_image = preprocess_input(expand_img)
    result = model.predict(pre_image).flatten()
    normalized = result / norm(result)
    return normalized

# Function to recommend similar images
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

# Upload image with a "Camera" button
uploaded_file = st.file_uploader("Choose an image")

# Display uploaded image and recommendations
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        resized_img = display_image.resize((200, 200))
        st.image(resized_img)
        st.text("Recommendations...")
        st.divider()

        features = extract_feature(os.path.join("uploads", uploaded_file.name), model)

        indices = recommend(features, feature_list)

        col1, col2, col3, col4, col5 = st.columns(5)
        try:
          base_dir = "C:\\Users\\Administrator\\Documents\\fashion_recommendation\\fashion-product-images-small\\images"+"\\"
          with col1:
            if not filenames[indices[0][0]].split("/")[-1] == uploaded_file.name:
               recommended_image1 = Image.open(base_dir+filenames[indices[0][0]].split("/")[-1])
               st.image(recommended_image1)
           

          with col2:
            recommended_image2 = Image.open(base_dir+filenames[indices[0][1]].split("/")[-1])
            st.image(recommended_image2)

          with col3:
            recommended_image3 = Image.open(base_dir+filenames[indices[0][2]].split("/")[-1])
            st.image(recommended_image3)

          with col4:
            recommended_image4 = Image.open(base_dir+filenames[indices[0][3]].split("/")[-1])
            st.image(recommended_image4)

          with col5:
            recommended_image5 = Image.open(base_dir+filenames[indices[0][4]].split("/")[-1])
            st.image(recommended_image5)
        except:
           pass    
    else:
        st.header("Error in file uploading")
st.caption('🤍Renjith')