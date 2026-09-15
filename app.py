%%writefile streamlit_app.py
import streamlit as st
import joblib
import pandas as pd

st.title('Delivery Delay Prediction App')

# Load the trained Logistic Regression model
logi_model = joblib.load('logi.sav')

st.sidebar.header('Input Features')

def user_input_features():
    delivery_distance = st.sidebar.slider('Delivery_Distance', 0.0, 100.0, 50.0)
    traffic_congestion = st.sidebar.slider('Traffic_Congestion', 1, 5, 3)
    weather_condition = st.sidebar.slider('Weather_Condition', 1, 3, 2)
    delivery_slot = st.sidebar.slider('Delivery_Slot', 1, 3, 2)
    driver_experience = st.sidebar.slider('Driver_Experience', 0, 20, 10)
    num_stops = st.sidebar.slider('Num_Stops', 1, 10, 5)
    vehicle_age = st.sidebar.slider('Vehicle_Age', 1, 10, 5)
    road_condition_score = st.sidebar.slider('Road_Condition_Score', 1, 5, 3)
    package_weight = st.sidebar.slider('Package_Weight', 0.0, 150.0, 75.0)
    fuel_efficiency = st.sidebar.slider('Fuel_Efficiency', 5.0, 25.0, 15.0)
    warehouse_processing_time = st.sidebar.slider('Warehouse_Processing_Time', 0, 100, 50)

    data = {'Delivery_Distance': delivery_distance,
            'Traffic_Congestion': traffic_congestion,
            'Weather_Condition': weather_condition,
            'Delivery_Slot': delivery_slot,
            'Driver_Experience': driver_experience,
            'Num_Stops': num_stops,
            'Vehicle_Age': vehicle_age,
            'Road_Condition_Score': road_condition_score,
            'Package_Weight': package_weight,
            'Fuel_Efficiency': fuel_efficiency,
            'Warehouse_Processing_Time': warehouse_processing_time}
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('User Input features')
st.write(df_input)

# Make prediction
prediction = logi_model.predict(df_input)
prediction_proba = logi_model.predict_proba(df_input)

st.subheader('Prediction')
if prediction[0] == 1:
    st.write('Delivery is likely to be Delayed')
else:
    st.write('Delivery is likely to be On-Time')

st.subheader('Prediction Probability (On-Time vs. Delayed)')
st.write(prediction_proba)
