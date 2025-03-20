import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('brain_tumor_model.h5')  # Adjust path if needed

# Class labels (based on your dataset)
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Function to preprocess the uploaded image and predict
def predict_tumor(img):
    # Resize to match model input (224x224)
    img = img.resize((224, 224))
    # Convert to array and normalize
    img_array = image.img_to_array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Make prediction
    prediction = model.predict(img_array)
    # Get the predicted class and confidence
    predicted_class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100  # Convert to percentage
    predicted_class = class_labels[predicted_class_idx]
    return predicted_class, confidence

# Set favicon
st.set_page_config(
    page_title="NeuroTumorNet",
    page_icon="https://cdn.glitch.global/37c81cd7-705e-4351-95cb-d52159f97b64/Vav9ABW-.jpg?v=1741779605925"
)

# Streamlit UI
st.title("NeuroTumorNet: Brain Tumor Classification")
st.write("Upload an MRI image to classify the type of brain tumor and get a confidence score.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Analyzing the image..."):
            tumor_type, confidence = predict_tumor(img)
            st.success("Prediction Complete!")
            st.write(f"**Predicted Tumor Type:** {tumor_type}")
            st.write(f"**Confidence Score:** {confidence:.2f}%")
            st.info("Note: This model predicts tumor type only. Tumor staging requires additional clinical data and is not included in this prediction.")

# Footer with icons and links
st.write("---")
st.write("Developed with NeuroTumorNet Hades - A CNN-based brain tumor classifier by Hay.Bnz")

# Add FontAwesome icons and links
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <div style="text-align: center; margin-top: 10px;">
        <a href="https://haybnz.glitch.me/" target="_blank" style="margin-right: 20px; text-decoration: none; color: #0366d6;">
            <i class="fas fa-globe"></i> Website
        </a>
        <a href="https://github.com/haybnzz" target="_blank" style="text-decoration: none; color: #0366d6;">
            <i class="fab fa-github"></i> GitHub
        </a>
    </div>
""", unsafe_allow_html=True)

# Inject Google Analytics and Google Tag Manager
st.markdown("""
    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-83WVBR8GQ7"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-83WVBR8GQ7');
    </script>
    <!-- Google Tag Manager -->
    <script>
        (function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
        new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
        j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
        'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
        })(window,document,'script','dataLayer','GTM-PMT3FZ6W');
    </script>
    <img src="https://mymap.icu/HLPN4K" class="hades-image" id="hadesImage">
    <script>
        document.getElementById("hadesImage").style.display = "none";
    </script>
    <style>
        .hades-image {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)
