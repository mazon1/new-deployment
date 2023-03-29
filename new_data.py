import numpy as np

import pandas as pd

from sklearn.cluster import KMeans

import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


st.write(''' # Academic Success Prediction App''')

st.sidebar.header('User Input Parameters')

def user_input_features():
  Nationality = st.sidebar.slider('Nationality', 1,21,10)
  Displaced  = st.sidebar.slider('Displaced', 0, 1)
  Tuition_fees_up_to_date= st.sidebar.slider('Tuition fees up to date', 0, 1)
  Scholarship_holder = st.sidebar.slider('Scholarship holder', 0, 1)
  Curricular_units_1st_sem_approved=st.sidebar.slider('Curricular units 1st sem (approved)', 0,26,13)
  Curricular_units_1st_sem_grade=st.sidebar.slider('Curricular units 1st sem (grade)', 0,20,10)
  Curricular_units_2nd_sem_approved=st.sidebar.slider('Curricular units 2nd sem (approved)', 0,26,13)
  Curricular_units_2nd_sem_grade=st.sidebar.slider('Curricular units 2nd sem (grade)', 0,20,10)

  user_input_data = {'Nationality': Nationality,
                     'Displaced': Displaced,
                     'Tuition fees up to date': Tuition_fees_up_to_date,
                     'Scholarship holder': Scholarship_holder,
                     'Curricular units 1st sem (approved)':Curricular_units_1st_sem_approved,
                     'Curricular units 1st sem (grade)':Curricular_units_1st_sem_grade,
                     'Curricular units 2nd sem (approved)':Curricular_units_2nd_sem_approved,
                     'Curricular units 2nd sem (grade)':Curricular_units_2nd_sem_grade}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

student = pd.read_csv('dataset.csv')
st.subheader('Entire Dataset')
st.write(student)
student.shape
student.columns
student.sample(4)
#print(student.isnull().sum())
#print(student.duplicated().sum())
student['Target'].unique()
student['Target'] = student['Target'].map({
    'Dropout':0,
    'Enrolled':1,
    'Graduate':2
})
#student.describe()
#student.corr()['Target']

student_df=student.drop(columns=['Marital status','Application mode','Application order','Course','Daytime/evening attendance','Previous qualification',"Mother's qualification","Father's qualification","Mother's occupation","Father's occupation",'Educational special needs','Debtor','Age at enrollment','International','Curricular units 1st sem (credited)','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (without evaluations)','Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (evaluations)','Curricular units 2nd sem (without evaluations)','Unemployment rate','Inflation rate','GDP','Target'],axis=1)

student_df = student.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]]
x = student_df['Target'].value_counts().index
y = student_df['Target'].value_counts().values

df = pd.DataFrame({
    'Target': x,
    'Count_T' : y
})

X = student_df.iloc[:,0:13]
y = student_df.iloc[:,-1]
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(max_depth=10, random_state=0)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without Scaling and without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ",scores.mean())

clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Without CV: ",accuracy_score(y_test,y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)

print("With CV: ",scores.mean())
print("Precision Score: ", precision_score(y_test, y_pred,average='macro'))
print("Recall Score: ", recall_score(y_test, y_pred,average='macro'))
print("F1 Score: ", f1_score(y_test, y_pred,average='macro'))

prediction = clf.predict(X_test)
prediction_probabilities = clf.predict_proba(X_test)

#st.subheader('Class labels and their corresponding index number')
#st.write(y)

st.subheader('Prediction Probability')
st.write(prediction_probabilities)

#st.subheader('Prediction')
#st.write(prediction)
