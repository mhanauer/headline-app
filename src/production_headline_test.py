import joblib
import os
from pyprojroot import here
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertModel
import torch

model_headline_stacker_bert = joblib.load("model_headline_stacker_bert.jlib")


def fun_bert_transform(new_text):
    # Load the tokenizer and model from the saved directory
    output_dir = "custom_headline_model"
    tokenizer_headline = DistilBertTokenizer.from_pretrained(output_dir)
    model_headline = DistilBertModel.from_pretrained(output_dir)

    # Tokenize the new text data and encode it as input IDs and attention masks
    tokens = tokenizer_headline.encode_plus(
        new_text,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
    )
    input_ids = torch.tensor(tokens["input_ids"]).unsqueeze(0)
    attention_masks = torch.tensor(tokens["attention_mask"]).unsqueeze(0)

    # Extract the features from the loaded DistilBERT model for the new text data
    with torch.no_grad():
        features_test = model_headline(input_ids, attention_mask=attention_masks)[0][
            :, 0, :
        ].numpy()
    x_test_out = pd.DataFrame(features_test)
    prob_out = model_headline_stacker_bert.predict(x_test_out.values)
    prob_out = np.where(prob_out < 0, 0, prob_out)
    prob_out = pd.DataFrame({"predicted_clicks": prob_out}).round(0)
    return prob_out["predicted_clicks"][0]

# Define Streamlit app
def app():
    st.title('Click Prediction Model')
    st.write('Type in a potential article title, email subject line, or social media post and get the predicted number of number clicks in a week.')
    st.write('The data are from OSF: https://osf.io/jd64p/files/osfstorage')

    headline = st.text_input('Enter the headline:')
    if st.button('Predict'):
        predicted_clicks = fun_bert_transform(headline)
        st.write(f'Predicted number of clicks: {predicted_clicks}')
        st.write('For this data the median clicks are 44 with a range of 0 to 811.  The median error is 15 clicks.')


app()
