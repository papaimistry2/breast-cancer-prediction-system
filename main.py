import streamlit as st
import numpy as np
import pickle

# Function to predict based on model
def predict(input_data):
    # Split the input data by commas and convert to float
    input_split_data = input_data.split(',')
    new_df = np.asarray(input_split_data).astype(np.float64)
    prediction = model.predict(new_df.reshape(1, -1))

    if prediction[0] == 1:
        st.write("<span style='color:red; font-weight:bold;'>Alert: You have a Breast Cancer Disease, check yourself by doctor.⚠️</span>", unsafe_allow_html=True)
    else:
        st.write("<span style='color:green; font-weight:bold;'>Don't Worry You don't have Breast Cancer Disease, Enjoy your life.😊</span>", unsafe_allow_html=True)


# Load the model
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))

# Streamlit app title
st.title("Breast Cancer Prediction System")



# Text input for breast cancer data
input_text = st.text_input("Enter Breast Cancer Data", "")

# Predict button
if st.button("Predict"):
    # Use the predict function directly with the input text
    try:
        predict(input_text)
    except ValueError:
        st.write("**Provide Proper Data Sets.**")


import streamlit as st
import pyperclip

def main():
    st.title("Here Are Some Data To Test This ML Model.")
    st.write("**(Click On Text To Copy.)**")
    st.text("The Copy System May Not Work In Some Mobile Devices.")

    # Define the text items
    text_items = ["-0.25693253, -0.1023531 ,  0.91607767, -0.03121146, -0.17413128,         2.33287657,  1.28710063,  0.83498609,  0.97009725,  1.61083696,         1.74987385,  0.58278358,  1.12042802,  0.55847914,  0.30008744,         0.56972638,  0.42132763,  0.14553046,  0.26701693, -0.05071461,         0.40980785,  0.5683385 ,  2.4279739 ,  0.59753719,  0.44819004,         4.04413686,  1.56843875,  0.98431948,  1.26303639,  2.74350941,         1.28908739", "-0.15858342, -0.17426394, -1.31587775, -0.15391903, -0.28161845,         0.43759352,  0.22158758, -0.12147652, -0.16760003,  0.54520436,         0.01079318, -0.63763544, -1.32296243, -0.48916675, -0.53984017,        -0.50511333, -0.1156695 , -0.13582661, -0.54066254, -0.2314578 ,        -0.39057552, -0.28983385, -1.41748481, -0.19694419, -0.36182533,         0.0880984 ,  0.05691742, -0.02380342, -0.37739621,  0.32121053,        -0.1317005", "-0.24842612,  1.84211602, -0.4389572 ,  1.67677878,  1.9703114 ,        -0.98108158, -0.5287112 , -0.0186444 ,  0.45024784, -0.00789214,        -0.97852283,  0.49835706, -0.92059392,  0.27642063,  0.86622356,        -0.73143343, -0.79604621, -0.47464392,  0.3539978 , -0.75630823,        -0.08110739,  1.74578351, -0.44056407,  1.47610602,  1.84279199,        -0.55417607, -0.44391322, -0.10225088,  1.03569383, -0.23696368,         0.25222467"]



    # Display the text items
    for text in text_items:
        if st.button(text):
            pyperclip.copy(text)
            st.success(f"Data Copied To Clipboard!")

if __name__ == "__main__":
    main()

