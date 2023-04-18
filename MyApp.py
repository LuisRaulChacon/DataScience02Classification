import streamlit as st
import joblib
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import torchvision.models as models

my_model = joblib.load("my_model.pkl")

st.write("""
# Classifier project
This model deploy lets you classify an image depending on the weather that image represents
""")

st.write("""
### The classes are:
#### Cloudy, Rain, Shine and Sunrise
""")

example = Image.open('example.jpg')
st.image(example)

st.write("""#### Please upload a jpg image""")

file = st.file_uploader(" Choose a file", type=["png", "jpg", "jpeg"])

if file is not None:
    data_kya = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_ANYCOLOR)

    img = Image.open(file)  # create an Image object
    st.image(img)  # show the image to the streamlit view

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize([0.4001, 0.4313, 0.4275], [0.0229, 0.0205, 0.0189])
    ])

    image = transform(img)

    # Cargar el modelo pre-entrenado
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()

    features = []

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        features.extend(output.view(output.size(0), -1).numpy())

    imageArray = np.array(features)

    predicted_label = my_model.predict(imageArray)  # the prediction
    number_class = predicted_label[0]

    if number_class == 0:
        prediction = "Cloudy"
    elif number_class == 1:
        prediction = "Rain"
    elif number_class == 2:
        prediction = "Shine"
    elif number_class == 3:
        prediction = "Sunrise"

    st.write('The class of the image is: ', prediction)
    st.write('Hope I could classify it correctly')
