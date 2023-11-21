import streamlit as st

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from help import VGG_11_CNN, process_image, prediction_result
import time

def main():
    classes = 36
    model = VGG_11_CNN(classes)
    model.load_state_dict(torch.load('bhanucha_hkongara_assignment2_part4.h5', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    

    st.set_option("deprecation.showfileUploaderEncoding", False)

    st.header("Deploying the model in local")
    st.write("**UBIT: bhanucha**")
    st.write("Here the images are transformed to 32x32, belongs to alphabets and numbers A-Z and 1-9")
    st.write("**About the model:** The model iam deploying belongs to VGG 11 configuration, implamented in part 4 of Assignement 2")
    img = st.file_uploader("Upload the image here", type=["jpeg", "jpg", "png"])

    try:
        img = Image.open(img)
        st.image(img) 
        img = process_image(img)
        img = torch.from_numpy(img).float()
        with torch.no_grad():
            output = model(img)

        prediction_result_dict = prediction_result(model, img)
        predicted_class = prediction_result_dict["class"]

        st.write(f'Predicted class: {predicted_class}')

        st.progress(100) 
        st.caching.clear_cache()

    except AttributeError:
         st.write("")

if __name__ == '__main__':
    # st.set_config_file(config_file="/streamlit/config.toml")
    main()