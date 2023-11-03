# -*- coding: utf-8 -*-
"""
Created on Tue May  3 21:00:45 2022

@author: Sahaanah
"""

import numpy as np
import pickle
import streamlit as st

#Loading the saved model
loaded_model = pickle.load(open("C:/Users/Sahaanah/Desktop/Python projects/trained_model.sav", 'rb'))


#creating a function for prediction

def cost_prediction(input_data):
   
    
    #Changing input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #Reshaping the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    
    return(prediction[0])
    
    
    
    
def main():
    
    #giving a webpage title
    st.title('Health Insurance Cost Prediction')
    
    
    #fetching the input data from the user
   
    
    age = st.text_input('Please specify your age')
    sex = st.text_input('Enter your sex [Male-0, Female-1]')
    bmi = st.text_input('Enter your Body Mass Index value')
    children = st.text_input('No: of Children')
    smoker = st.text_input('Do you have the habit of smoking? [Yes-1, No-0]')
    
    #code for prediction
    FinalCost = ''
    
    #creating a button for prediction
    
    if st.button('Insurance Cost'):
        FinalCost = cost_prediction([age,sex,bmi,children,smoker])
        
        
    st.success('The Insurance cost is Rs. {}' .format(FinalCost))

    
    
if __name__ == '__main__':
    main()