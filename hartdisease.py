
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('/content/heart_disease.csv')

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

features = ['Age', 'Cholesterol', 'Blood Pressure']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X = df[['Age', 'Gender', 'Cholesterol', 'Blood Pressure']]
y = df['Heart Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()
