import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('IRIS Flower.csv')

# Data preprocessing
label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

X = data.drop('species', axis=1)
y = data['species']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Building the model
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluating the model
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Output the feature importance
importances = classifier.feature_importances_
feature_names = X.columns
feature_importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
print("Feature importances:", feature_importance_dict)