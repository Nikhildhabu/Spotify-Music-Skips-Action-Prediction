# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st

#pickle_in = open("lgbm_model_TestAcc.pkl","rb")
pickle_in = open("rfc_model_TestAcc.pkl","rb")
rfc=pickle.load(pickle_in)

col1, mid, col2 = st.beta_columns([1,1,20])
with col1:
    st.image('spotify.png', width=90)
with col2:
    st.write("""
# Predicting Spotify Track Skips
This app predicts the **Spotify Track Skips**
""")
    st.sidebar.header('User Input Parameters')



def user_input_features():
    session_position = st.sidebar.text_input('session_position',0)
    session_length = st.sidebar.text_input('session_length', 0, 1, 1)
    no_pause_before_play = st.sidebar.text_input('no_pause_before_play',0,1,1)
    short_pause_before_play = st.sidebar.text_input('short_pause_before_play', 0.0,1,1)
    long_pause_before_play = st.sidebar.text_input('long_pause_before_play',0.0,1,1)
    hist_user_behavior_n_seekfwd = st.sidebar.text_input('hist_user_behavior_n_seekfwd', -0.5,40,0.1)
    hist_user_behavior_n_seekback = st.sidebar.text_input('hist_user_behavior_n_seekback', -0.5,40,0.1)
    hour_of_day = st.sidebar.text_input('hour_of_day', 0)
    context_type_catalog = st.sidebar.text_input('context_type_catalog',  0, 1, 1)
    context_type_charts = st.sidebar.text_input('context_type_charts',  0, 1, 1)
    context_type_editorial_playlist = st.sidebar.text_input('context_type_editorial_playlist', 0, 1 )
    context_type_personalized_playlist = st.sidebar.text_input('context_type_personalized_playlist', 0.1,1 )
    context_type_radio = st.sidebar.text_input('context_type_radio', 0,1,1)
    context_type_user_collection = st.sidebar.text_input('context_type_user_collection', 0,1,1)
    hist_user_behavior_reason_start_appload = st.sidebar.text_input('hist_user_behavior_reason_start_appload', 0,1,1)
    hist_user_behavior_reason_start_backbtn = st.sidebar.text_input('hist_user_behavior_reason_start_backbtn', 0,1,1)
    hist_user_behavior_reason_start_clickrow = st.sidebar.text_input('hist_user_behavior_reason_start_clickrow', 0,1,1)
    hist_user_behavior_reason_start_endplay = st.sidebar.text_input('hist_user_behavior_reason_start_endplay', 0,1,1)
    hist_user_behavior_reason_start_fwdbtn = st.sidebar.text_input('hist_user_behavior_reason_start_fwdbtn', 0,1,1)
    hist_user_behavior_reason_start_playbtn = st.sidebar.text_input('hist_user_behavior_reason_start_playbtn', 0,1,1)
    hist_user_behavior_reason_start_remote= st.sidebar.text_input('hist_user_behavior_reason_start_remote', 0,1,1)
    hist_user_behavior_reason_start_trackdone= st.sidebar.text_input('hist_user_behavior_reason_start_trackdone', 0,1,1)
    hist_user_behavior_reason_start_trackerror= st.sidebar.text_input('hist_user_behavior_reason_start_trackerror', 0,1,1)


    data = {'session_position':session_position,
            'session_length':session_length,
            'no_pause_before_play':no_pause_before_play,
            'short_pause_before_play':short_pause_before_play,
            'long_pause_before_play':long_pause_before_play,
          'hist_user_behavior_n_seekfwd':hist_user_behavior_n_seekfwd,
            'hist_user_behavior_n_seekback':hist_user_behavior_n_seekback,
            'hour_of_day':hour_of_day,
            'context_type_catalog':context_type_catalog,
            'context_type_charts':context_type_charts,
            'context_type_editorial_playlist':context_type_editorial_playlist,
          'context_type_personalized_playlist':context_type_personalized_playlist,
            'context_type_radio':context_type_radio,
            'context_type_user_collection':context_type_user_collection,
            'hist_user_behavior_reason_start_appload':hist_user_behavior_reason_start_appload,
            'hist_user_behavior_reason_start_backbtn':hist_user_behavior_reason_start_backbtn,
            'hist_user_behavior_reason_start_clickrow':hist_user_behavior_reason_start_clickrow,
            'hist_user_behavior_reason_start_endplay':hist_user_behavior_reason_start_endplay,
          'hist_user_behavior_reason_start_fwdbtn':hist_user_behavior_reason_start_fwdbtn,
            'hist_user_behavior_reason_start_playbtn':hist_user_behavior_reason_start_playbtn,
            'hist_user_behavior_reason_start_remote':hist_user_behavior_reason_start_remote,
            'hist_user_behavior_reason_start_trackdone':hist_user_behavior_reason_start_trackdone,
            'hist_user_behavior_reason_start_trackerror':hist_user_behavior_reason_start_trackerror}


    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Class labels and their corresponding index number')
st.write(df)

col1,mid, col2 = st.beta_columns([20,1,20])
with col1:
    st.subheader('Prediction')
    prediction = rfc.predict(df)
    st.write(prediction)
with col2:
    st.subheader('Prediction Probability')
    predict_proba=rfc.predict_proba(df)
    st.write(predict_proba)

st.markdown("<h3 style='text-align: center;'>Author: Nikhil Dhabu</h3>", unsafe_allow_html=True)


