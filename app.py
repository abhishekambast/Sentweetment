



import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D,Dropout

from tensorflow.keras import layers


train=pd.read_csv("archive/training.csv")

def get_tweet(data):
    tweets=data['text'].tolist()
    labels=data['label'].tolist()
    
    return np.array(tweets),np.array(labels)

tweets,labels=get_tweet(train)

def get_seq(tweets,tokenizer):
    maxlen=40
    
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    train_sequences = tokenizer.texts_to_sequences(tweets)
    train_padded = pad_sequences(train_sequences, padding='post', truncating='post', maxlen=maxlen)
    return np.array(train_padded)


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(num_words=10000,oov_token='<UNK>',)
tokenizer.fit_on_texts(tweets)



class Emotion(keras.Model):
    

    def __init__(self):
        super().__init__()
        self.embedding = Embedding(10000,16,input_length=40)
        self.LSTM1 = layers.Bidirectional(layers.LSTM(20,  return_sequences=True))
        self.LSTM2 = layers.Bidirectional(layers.LSTM(20))
        self.dropout=Dropout(0.3)
        self.dense = layers.Dense(6,activation='softmax')

    def call(self, inputs,training=False):
        x = self.embedding(inputs)
        y=self.LSTM1(x)
        z= self.LSTM2(y)
        if training:
            z = self.dropout(z,training=training)
        result = self.dense(z)
        return result
    




def check_emotion(seq1):
    model=Emotion()
    dummy_input = tf.zeros((1, 40))  # Adjust the input shape if necessary
    model(dummy_input)
    model.load_weights("emotion_model_weights.h5")
    k=np.array(model.predict(seq1)).argmax()
    if(k==0):
        return "It feels like you are sad, take a song and make it better"
    if(k==1):
        return "It feels like you are feeling happy, be forever like this"
    if(k==2):
        return "It feels like u r feeling love, amazing"
    if(k==3):
        return "It feels like you are angry, please calm down and don't take any decision when u are angry!"
    if(k==4):
        return "It feels like u r feeing feared, it's ok but do not let it over you for long time"
    if(k==5):
        return "It feels like you are feeling surprised!"
    
    
st.title("Tweet Emotion Recognition")
st.write("Enter a tweet to predict the emotion:")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpaperset.com/w/full/1/6/d/89748.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
    
@st.cache(allow_output_mutation=True)
def load_curr_model():
    with st.spinner('Model is being loaded..'):
        model=Emotion()
        dummy_input = tf.zeros((1, 40))
        model(dummy_input)
        model.load_weights("emotion_model_weights.h5")
        return model
    
   

    # Get user input
statement = st.text_input("").lower()
    



    # Make a prediction when the user submits the input
if st.button("Predict"):
    if statement.strip() != "":
            # Predict the emotion
        statement=[statement]
        seq=np.array(statement)
        seq1=get_seq(seq,tokenizer)
        predicted_emotion = check_emotion(seq1)
        st.success(f"Predicted Emotion: {predicted_emotion}")
    else:
        st.warning("Please enter some text.")

# Run the app




st.write("\n\n By Abhishek Ambast.")






