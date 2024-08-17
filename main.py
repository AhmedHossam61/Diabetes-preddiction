import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
# loading the diabetes dataset to a pandas DataFrame
df = pd.read_csv('diabetes.csv') 


st.title('Diabetes checkup')


st.subheader("Training data")
st.write(df.describe())

x = df.drop(['Outcome'] , axis=1)
y = df.iloc[ : , -1] #all rows and only last column

st.subheader('*Visualization:*')
st.bar_chart(df)

st.subheader("X-data")
st.write(x.head())

st.subheader("y-data")
st.write(y.head())

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.2, random_state=0)


#This function is for taking the inputs from the user
def parameters():
    Pregnancies = st.sidebar.slider('Pregnancies' , 0 , 17 , 3) 
    glucose     = st.sidebar.slider('Glucose', 0 , 200 ,120 )
    bd          = st.sidebar.slider('BloodPressure' , 0 ,122 ,70)
    skinTknss   = st.sidebar.slider('SkinThickness' , 0 , 99 , 23 )
    insulin     = st.sidebar.slider('Insulin' , 0 , 846 , 30)
    bmi         = st.sidebar.slider('BMI' )
    DPf         = st.sidebar.slider('DiabetesPedigreeFunction' , 0.078 ,2.42 ,0.3725 )
    age         = st.sidebar.slider('Age' , 21 , 81 , 29)

    user_report = {
        'Pregnancies': Pregnancies,
        'Glucose':glucose,
        'BloodPressure':bd,
        'SkinThickness':skinTknss,
        'Insulin':insulin,
        'BMI':bmi,
        'DiabetesPedigreeFunction':DPf,
        'Age':age
    }
    report_data = pd.DataFrame(user_report , index = [0])
    return report_data

user_data = parameters()

st.header("1- Random Forest Classifier")
rf = RandomForestClassifier()
rf.fit(X_train , Y_train)

st.subheader('Accuaracy:')
st.write(str(accuracy_score(Y_test , rf.predict(X_test))*100)+ '%')

user_result = rf.predict(user_data)
st.subheader("your result:")
output = ""
if user_result[0] == 0:
    output = "You are healthy"
else:
    output = "You are sick"

st.write(output)



st.header("2- Support Vector Machine")
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train , Y_train)
st.subheader('Accuaracy:')
st.write(str(accuracy_score(Y_test , classifier.predict(X_test))*100)+ '%')

user_result = classifier.predict(user_data)
st.subheader("your result:")
output = ""
if user_result[0] == 0:
    output = "You are healthy"
else:
    output = "You are sick"

st.write(output)