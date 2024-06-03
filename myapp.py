import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

st.write("""
# Simple House Price Prediction
by Zulfa Fakaha 21.11.4337

Predicting house prices based on features such as square footage, number of bathrooms, bedrooms, and floors.
""")

# Load the dataset
df = pd.read_csv("DataHargaRumah.csv")

# Feature engineering
# df['age'] = 2024 - df['year_built']
# df['bath_bed_ratio'] = df['bathrooms'] / df['bedrooms']

# Selecting features and target
# X = df[['sqft_living', 'bathrooms', 'bedrooms', 'floors', 'age', 'bath_bed_ratio']]
X = df[['sqft_living', 'bathrooms', 'bedrooms', 'floors']]
y = df['price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
def predict_price(sqft_living, bathrooms, bedrooms, floors):
    input_data = np.array([[sqft_living, bathrooms, bedrooms, floors]])
    predicted_price = model.predict(input_data)[0]
    return predicted_price

# st.sidebar.header('Input Features')

sqft_living = st.sidebar.slider('Square Footage', min_value=100, max_value=10000, value=1000, step=50)
bathrooms = st.sidebar.slider('Number of Bathrooms', min_value=1, max_value=10, value=2, step=1)
bedrooms = st.sidebar.slider('Number of Bedrooms', min_value=1, max_value=10, value=4, step=1)
floors = st.sidebar.slider('Number of Floors', min_value=1, max_value=5, value=1, step=1)

# predicted_price = predict_price(sqft_living, bathrooms, bedrooms, floors)

st.subheader('Predicted House Price:')
prediksi_harga_tunggal = predict_price(sqft_living, bathrooms, bedrooms, floors) * 16045
st.write(f"Rp {prediksi_harga_tunggal:,.2f}".replace(",", "."))

st.write(f"${predict_price(sqft_living, bathrooms, bedrooms, floors):,.2f}")
st.write(f"Rp{predict_price(sqft_living, bathrooms, bedrooms, floors)*16045:,.2f}".replace(",", "."))


