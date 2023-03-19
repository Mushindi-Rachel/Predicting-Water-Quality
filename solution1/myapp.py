import joblib
import streamlit as st
import numpy as np
import os

location = ''
fullpath = os.path.join(location, 'model.pkl')

model = joblib.load(open(fullpath, 'rb'))


def water_potability_prediction(input_data):
    input_as_array = np.array(input_data).reshape(1,-1)
    prediction = model.predict(input_as_array)[0]
    return prediction


def main():
    st.set_page_config(page_title='Water Quality Prediction', page_icon=':potable_water:')
    st.title('Water Quality Prediction')
    st.write('This app predicts whether water is potable.')
    
    st.subheader('Water Quality Parameters')
    
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input('Hardness', min_value=0.0, value=100.0, step=1.0)
    solids = st.number_input('Solids', min_value=0.0, value=200.0, step=1.0)
    chloramines = st.number_input('Chloramines', min_value=0.0, value=5.0, step=0.1)
    sulfate = st.number_input('Sulfate', min_value=0.0, value=300.0, step=1.0)
    conductivity = st.number_input('Conductivity', min_value=0.0, value=400.0, step=1.0)
    organic_carbon = st.number_input('Organic Carbon', min_value=0.0, value=10.0, step=0.1)
    trihalomethanes = st.number_input('Trihalomethanes', min_value=0.0, value=50.0, step=0.1)
    turbidity = st.number_input('Turbidity', min_value=0.0, value=5.0, step=0.1)
    
    if st.button('Check Potability'):
        input_data = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
        prediction = water_potability_prediction(input_data)
        if prediction == 0:
            st.error('The water is not potable.')
        else:
            st.success('The water is potable.')
    
    st.write('---')
    st.write('Model accuracy on test data: 0.89')
    
    
    
if __name__ == '__main__':
    main()
