import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Student Enrollment Prediction")

st.sidebar.header("Enter Student Data")

def user_input_features():
    dataset = pd.read_csv("dataset.csv")
    Tuition_fees_up_to_date = st.sidebar.slider('Tuition fees up to date', 0, 1)
    Scholarship_holder = st.sidebar.slider('Scholarship holder', 0, 1)
    Marital_Status = st.sidebar.selectbox("Marital status", dataset["Marital status"].unique())
    Gender = st.sidebar.selectbox("Gender", dataset["Gender"].unique())
    Age_at_enrollment = st.sidebar.slider('Age at enrollment', 18, 55, 36)
    International = st.sidebar.slider('International', 0, 1)

    user_input_data = {'Tuition fees up to date': Tuition_fees_up_to_date,
                     'Scholarship holder': Scholarship_holder,
                     'Marital status': Marital_Status,
                     'Gender': Gender,
                     'Age at enrollment': Age_at_enrollment,
                     'International': International}

    features = pd.DataFrame(user_input_data, index=[0])

    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

student = pd.read_csv('dataset.csv')
student['Target'] = student['Target'].map({
    'Dropout': 0,
    'Enrolled': 1,
    'Graduate': 2
})
student_df = student.drop(columns=['Nacionality', 'Displaced', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance', 'Previous qualification', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", 'Educational special needs', 'Debtor', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP', 'Target'], axis=1)

st.subheader('Dataset')
st.write(student_df)

X = student_df
y = student['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
st.write("Accuracy:", accuracy_score(y_test, y_pred))

prediction = clf.predict(df)
prediction_probabilities = clf.predict_proba(df)

status = ''
if prediction[0] == 0:
    status = 'Dropout'
elif prediction[0] == 1:
    status = 'Enrolled'
else:
    status = 'Graduate'

st.subheader('Prediction')
st.write('The student is likely to', status)

st.subheader('Prediction Probability')
st.write(prediction_probabilities)
