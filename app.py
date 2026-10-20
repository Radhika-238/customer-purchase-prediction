import streamlit as st
import pandas as pd
import joblib

model = joblib.load('customer_purchase_prediction.pkl')

st.title('Customer Purchase Prediction')
st.write("Predict whether an online shopper is likely to make a purchase based on their browsing behavior, engagement metrics, and visitor information using a Machine Learning model.")

st.subheader('Browsing Activity')

administrative = st.number_input(
    'Administrative Pages Visited', 
    min_value = 0,
    value = 0,
    step = 1
)

administrative_duration = st.number_input(
    'administrative_duration(seconds)', 
    min_value = 0.0,
    value = 0.0,
    step = 1.0
)

informational = st.number_input(
    'Informational Pages Visited', 
    min_value = 0,
    value = 0,
    step = 1
)

informational_duration = st.number_input(
    'Informational Duration (seconds)', 
    min_value = 0.0,
    value = 0.0,
    step = 1.0
)

product_related = st.number_input(
    'Product Pages Visited', 
    min_value = 0,
    value = 0,
    step = 1
)

product_related_duration = st.number_input(
    'Product Related Duration (seconds)', 
    min_value = 0.0,
    value = 0.0,
    step = 1.0
)

st.subheader('Engagement Metrics')
bounce_rates = st.number_input(
    'Bounce Rate', 
    min_value = 0.0,
    value = 0.0,
    step = 0.01,
    max_value = 1.0
)

exit_rates = st.number_input(
    'Exit Rate', 
    min_value = 0.0,
    value = 0.0,
    max_value = 1.0
)

page_values = st.number_input(
    'Page Values', 
    min_value = 0.0,
    value = 0.0,
    step = 1.0
)

special_day = st.number_input(
    'Special Day Score', 
    min_value = 0.0,
    value = 0.0,
    max_value = 1.0,
    step = 0.2
)

st.subheader('Visitor Details')

month = st.selectbox(
    'Month',
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul',
     'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
)

operating_system = st.selectbox(
    'Operating System',
    list(range(1, 9))
)

browser = st.selectbox(
    'Browser',
    list(range(1, 14))
)

region = st.selectbox(
    'Region',
    list(range(1, 10))
)

traffic_type = st.selectbox(
    'Traffic Type',
    list(range(1, 21))
)

visitor_type = st.selectbox(
    'Visitor Type',
    ['Returning_Visitor', 'New_Visitor', 'Other']
)

weekend = st.checkbox(
    'Weekend',
    value=False
)

input_data = pd.DataFrame({
    'Administrative': [administrative],
    'Administrative_Duration': [administrative_duration],
    'Informational': [informational],
    'Informational_Duration': [informational_duration],
    'ProductRelated': [product_related],
    'ProductRelated_Duration': [product_related_duration],
    'BounceRates': [bounce_rates],
    'ExitRates': [exit_rates],
    'PageValues': [page_values],
    'SpecialDay': [special_day],
    'Month': [month],
    'OperatingSystems': [operating_system],
    'Browser': [browser],
    'Region': [region],
    'TrafficType': [traffic_type],
    'VisitorType': [visitor_type],
    'Weekend': [weekend]
})

if st.button('Predict Purchase'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success('Customer is likely to make a purchase')
    else:
        st.error('Customer is unlikely to make a purchase')
        