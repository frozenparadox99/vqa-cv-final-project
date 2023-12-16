

import streamlit as st
import requests
import os

st.title("Visual Question Answering Client")

# Input for image URL
image_url = st.text_input("Enter the image URL:")
if image_url and os.path.isfile(image_url):
    st.image(image_url, use_column_width=True)
else:
    st.write("Please enter a valid image path.")

# Question input
question = st.text_input("Enter your question:")

# On 'Predict' button click
if st.button('Predict'):
    if image_url and question:
        # Prepare the request payload
        payload = {'image_url': image_url, 'question': question}

        # Send POST request to Flask server
        response = requests.post("http://127.0.0.1:5000/", json=payload)

        if response.status_code == 200:
            answer = response.json().get('answer')
            st.write(f"Answer: {answer}")
        else:
            st.write("Error in prediction:", response.text)
    else:
        st.write("Please enter an image URL and a question.")

