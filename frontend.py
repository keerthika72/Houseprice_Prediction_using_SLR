#Importing the required libraries:
import streamlit as st  
import pickle
import numpy as np

#loading the pickle file 
model = pickle.load(open(r"C:\Users\monik\Downloads\house_price_prediction\house price_prdiction.pkl","rb"))

st.title("House Price Prediction App ")

#Adding a brief description about the app:
st.write("This app predicts the price of the house based on the area per square feet present.")

#sepcify the input values each and every parameters like the min , max value
area_per_sqft= st.number_input("Enter the area per square feet:-",min_value=0.0,max_value=500000.0,value=1.0,step=1.0)

if st.button("Predict House Price"): 
    area_input=np.array([[area_per_sqft]])
    prediction=model.predict(area_input) #This will help to show the predicted value .
    
    st.success(f"The estimated Price for {area_input} years of experience is: **${prediction[0]:,.2f}**") #helps to print out the predicted output.
    
st.write("The model was trained using the dataset of house price and its corresponding area per square feet.")

