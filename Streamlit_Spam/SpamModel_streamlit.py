import pickle
import streamlit as st 
from  win32com.client import Dispatch
import string


def speak(text):
	speak=Dispatch(('SAPI.SpVoice'))
	speak.Speak(text)
def process_mail(text):
    nopunctuation=[char for char in text if char not in string.punctuation]
    nopunctuation=''.join(nopunctuation)
    cleanwords=[word for word in nopunctuation.split()]
    return cleanwords
classifier=pickle.load(open("Spam.pkl","rb"))
vectorizer=pickle.load(open("vectorizer.pkl","rb"))

def main():
	speak("Spam Classifier By Sabri Mahmoud")
	st.title("Spam Classifier By Sabri Mahmoud")
	speak("First Step with NLP")
	speak("Please wait")
	st.subheader("First Step with NLP")
	msg=st.text_input("Enter a text")
	if st.button("Predict"):
		 data=vectorizer.transform([msg]).toarray()
		 prediction=classifier.predict(data)
		 if prediction[0]==1:
		 	spam="This is a Spam"
		 	st.error(spam)
		 	speak(spam)
		 else: 
		 	notSpam="This is not a Spam"
		 	st.success(notSpam)
		 	speak(notSpam)
main()