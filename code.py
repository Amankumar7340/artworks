import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('heart.csv')

print("Missing values per column:\n", df.isnull().sum())

df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)

print("\nDataset overview:\n", df.info())

print("\nStatistical summary:\n", df.describe())

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show(block=False)  

plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title('Target Variable Distribution')
plt.show(block=False) 

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=False) 

new_data = pd.DataFrame([
    {'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233, 'fbs': 1, 'restecg': 0,
     'thalach': 150, 'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1}
])
new_prediction = model.predict(new_data)
print("\nPrediction for new data:", new_prediction)

plt.show()  