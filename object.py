import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🔍"
)

st.title("🔍 AI Object Detection")

@st.cache_resource
def load_model():
    return pipeline(
        "object-detection",
        model="facebook/detr-resnet-50"
    )

detector = load_model()

image = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

if image:
    img = Image.open(image)

    st.image(img, caption="Uploaded Image")

    if st.button("🔍 Detect Objects"):

        with st.spinner("Detecting objects..."):
            result = detector(img)

        st.subheader("Detected Objects")

        for obj in result:
            st.write(
                f"{obj['label']}"
            )
