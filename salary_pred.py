import pandas as pd
import streamlit as st
df=pd.read_csv("salary_predict_dataset.csv")
df.head()
x=df.iloc[:,:4]
y=df.iloc[:,4:]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
st.header("Salary Prediction")
st.sidebar.header("User Input")
exp=st.sidebar.slider("Numbers of Experience",0,15,4)
cgpa=st.sidebar.slider("CGPA Score",0,10,5)
age=st.sidebar.slider("Age",18,60,20)
int_scr=st.sidebar.slider("Interview Score",0,100,50)

y_pred1=lr.predict([[exp,cgpa,age,int_scr]])
y_pred2=lr.predict(x_test)
st.subheader("Predicted Salary")
st.write(y_pred1)
from sklearn.metrics import r2_score
st.subheader("Accuracy")
st.write(r2_score(y_test,y_pred2))


#After running the program type streamlit run File_name.py in the terminal


