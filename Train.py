import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv(r'C:\\Users\\Hxtreme\\PycharmProjects\\MLProject\\data\\Bank Customer Churn.csv')
print(df.head())

print(df.describe())
print(df.shape)
print(df.isnull().sum())

df.drop('customer_id', axis=1, inplace=True)

label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

x = df.drop('churn', axis=1)
y = df['churn']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)
accuracy = accuracy_score(y_test,y_pred_rf)
print("Accuracy Score : ",accuracy)

joblib.dump(rf,"rf.pkl")

loaded_model = joblib.load("rf.pkl")
