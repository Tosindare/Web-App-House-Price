# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:51:13 2020

@author: TosinOja
"""
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

house_data = pd.read_csv('kc_house_data.csv', encoding = 'ISO-8859-1' )


    

#"""House price Prediction App"""
st.write("""
    # House Price Prediction App
    *Washington DC, USA*
    """)
    
html_temp = """
    <div style = "background - color: #f0f0f5; padding: 5px">
    <h3 style="color:#666666;text-align:left; line-height: 1.5">
    <p>This Web App will predict house 
    price once the following (16) parameters are inputed.<br> 
    This is based on Deep learning 
    algorithms with data from 2014/15 House Sales in King County, Washington DC.</p></h3>
    </div>
    
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.sidebar.header('User Input Parameters') 
    
if st.checkbox('Show Summary of Dataset'):
    st.write(house_data.describe())
    

#@st.cache      
def user_input_parameters():         
    bedrooms = st.sidebar.slider("1. No of bedrooms?", 0, 12, 4)
    #st.write(""" **You selected** """, bedrooms, """**bedrooms**""")
    
    bathrooms = st.sidebar.slider("2. No of bathrooms?", 0, 15, 5)
    #st.write(""" **You selected** """, bathrooms, """**bathrooms**""")
    sqft_living = st.sidebar.slider("3. Square footage of the house?", 500, 15000,2000)
    #st.write(""" **You chose** """, sqft_living,"""**Square fts**""")
        
    sqft_lot = st.sidebar.slider("4. Square footage of the lot?",500, 170000,1200 )
    #st.write(""" **You wrote** """, sqft_lot,"""**Square fts**""")
    floors = st.sidebar.slider("5. No of floors?", 0, 5, 3)
    #st.write(""" **You selected** """, floors,"""**floors**""")
    #views = st.sidebar.slider("6. No of viewings of the house?", 0,10,0)
    #st.write(""" **You selected** """, views,"""**views**""")
    condition = st.sidebar.slider("7. Overall condition? (1 indicates worn out property and 5 excellent)", 0,5,3)
    #st.write(""" **You selected** """, condition,"""**as the overall condition of the house**""")
    grade = st.sidebar.slider("8. Overall grade based on King County grading system? (1 poor ,13 excellent)", 0,13,6)
    #st.write(""" **You selected grade** """, grade)
    sqft_above = st.sidebar.slider("9. Square footage above basement?", 200, 12000, 5000)
    #st.write(""" **You chose** """, sqft_abovebsmt,"""**Square fts**""")
    sqft_basement = st.sidebar.slider("10. Square footage of the basement?", 0, 7000, 2500)
    #st.write(""" **You chose** """, sqft_basement,"""**Square fts**""")
    yr_built = st.sidebar.slider("11. Year Built?", 1900,2019, 2009)
    #st.write(""" **You selected** """, yr_built)
    yr_renovated = st.sidebar.radio('12. Year renovated?',('Known', 'Unknown'))
    if yr_renovated == 'Unknown':
        yr_renovated = 0
    else:
        yr_renovated = st.sidebar.slider("Year Renovated?", 1900,2019, 2010)

    zipcode = st.sidebar.slider("13. Zipcode of the house?", 98001,98288, 98250)
    #st.write(""" **You selected** """, zipcode)
    lat = st.sidebar.slider("14. Location of House (lattitude)?", 47.000100, 47.800600, 47.560053, 0.000001, "%g")
    long = st.sidebar.slider("15. Location of House (longitude)?", -122.6000000,-121.300500, -122.213896, 0.000001,"%g")
    sqft_living15 = st.sidebar.slider("16. Square footage of the house in 2015?", 200, 12000, 3500)
    #st.write(""" **You chose** """, sqft_living15,"""**Square fts**""")
    sqft_lot15 = st.sidebar.slider("17. Square footage of the lot in 2015?", 200, 12000, 3700)
    #st.write(""" **You chose** """, sqft_lot15,"""**Square fts**""")
    waterfront = st.sidebar.radio('18. House has Waterfront View?',('Yes', 'No'))
    if waterfront == 'Yes':
        waterfront = 1
    else:
        waterfront = 0 
    
    features = {'bedrooms' : bedrooms, 'bathrooms' : bathrooms, 
                'sqft_living'  : sqft_living, 'sqft_lot' : sqft_lot,
                'floors' : floors, 'waterfront' : waterfront,
                'condition' : condition, 'grade' : grade, 
                'sqft_above' : sqft_above,'sqft_basement': sqft_basement, 
                'yr_built' : yr_built, 'yr_renovated': yr_renovated, 'zipcode' : zipcode, 
                'lat' : lat, 'long' : long, 'sqft_living15' : sqft_living15,
                'sqft_lot15' : sqft_lot15}
    
    feat = pd.DataFrame(features, index=[0])
    
    return feat
    

df = user_input_parameters()

st.subheader('User Input parameters')
st.write(df)    


#model_rf = joblib.load("random_forest_regression_model.joblib")

model_ann = tf.keras.models.load_model("ann_model.hdf5")
 
# def predict_rf():
#     prediction = model_rf.predict(df)
#     pred = np.around(prediction, 2)
#     return float(pred)   

# house_price_1 = predict_rf()

st.text("")



from sklearn.preprocessing import StandardScaler


#scalerX = joblib.load('scaler_x1.gz')
scalerY = joblib.load('scaler_y1.gz')
scaler = standardScaler()
X_df = scaler.fit_transform(df)


def predict_ann():
    prediction = model_ann.predict(X_df)
    prediction_org = scalerY.inverse_transform(prediction)
    predict = np.around(prediction_org, 2)
    return float(predict)   

house_price_2 = predict_ann()

st.text("")
if st.button('PREDICT PRICE'):
    st.write("**$**", house_price_2, " -*based on Deep Learning Algorithm (80% accuracy)*")

url = '[SOURCE CODE](https://github.com/Tosindare/Web-App-House-Price)'

st.markdown(url, unsafe_allow_html=True)    
#st.subheader(')


#if __name__ =='__main__':
    #main()
              
    
