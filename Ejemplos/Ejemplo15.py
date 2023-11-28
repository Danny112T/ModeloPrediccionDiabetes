import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

y_pred = knn_model.predict(X_test)

print(f'Confusion Matrix:\n{confusion_matrix(Y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(Y_test, y_pred)}')