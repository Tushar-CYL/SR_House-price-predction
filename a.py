import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('predction_flat.csv')

# Train the model
reg = LinearRegression()
reg.fit(df[['area']], df['price'])

# Define the Streamlit app
def main():
    st.title('House Price Prediction')

    area_input = st.number_input('Enter the area:', min_value=0.0)
    if st.button('Predict'):
        prediction = reg.predict(np.array(area_input).reshape(1, -1))
        st.success(f'Predicted price for {area_input} sqft: ${prediction[0]:,.2f}')
        st.balloons()
    

if __name__ == '__main__':
    main()
