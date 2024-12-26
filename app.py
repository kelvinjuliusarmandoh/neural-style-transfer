import streamlit as st
import torch
from PIL import Image
from modulars.engine import engine
from modulars.model import VGGModified


def app():
    st.title("Neural Style Transfer Web App")
    st.subheader("Combined the content images and style images, then generate stylized images.")

    content_image = st.file_uploader("Content Image", type=['jpg', 'jpeg', 'png'])
    style_image = st.file_uploader("Style Image", type=['jpg', 'jpeg', 'png'])

    
    # Parameters
    with st.sidebar:
        st.title("Setup Your Parameters")
        st.info("Setting up your parameter to look the difference !")
        IMAGE_SIZE = st.slider("Image Size", min_value=32, max_value=224, step=32)
        TOTAL_STEPS = st.slider("Total Steps", min_value=0, max_value=3000, step=100)
        ALPHA = st.slider("Alpha Constant", min_value=1, max_value=10)
        BETHA = st.slider("Betha Constant", min_value=0, max_value=10000, step=100)
        LEARNING_RATE = 0.001

        process_button = st.button("Process")

    if content_image and style_image:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        content_image_raw = Image.open(content_image).resize((224, 224))
        style_image_raw = Image.open(style_image).resize((224, 224))

        # Create columns layout
        column1, column2, column3 = st.columns(3, gap="medium")

        with column1:
            st.header("Content")
            st.image(content_image_raw)

        with column2:
            st.header("Style")
            st.image(style_image_raw)

        # Load model
        vgg = VGGModified().to(device).eval()
        
        if process_button:
            # Run the process 
            final_generated_image = engine(
                                        vgg_model=vgg,
                                        content_image=content_image,
                                        style_image=style_image,
                                        total_steps=TOTAL_STEPS,
                                        image_size=IMAGE_SIZE,
                                        alpha=ALPHA,
                                        betha=BETHA,
                                        learning_rate=LEARNING_RATE,
                                        device=device).permute(1, 2, 0).detach().cpu().numpy()
            
            final_generated_image = (final_generated_image - final_generated_image.min()) / (final_generated_image.max() - final_generated_image.min())

            with column3:
                st.header("Result")
                st.image(final_generated_image)


if __name__ == '__main__':
    app()