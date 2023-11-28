import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, svm
import matplotlib.pyplot as plt

df = pd.read_csv(
    "/Users/danny/Library/Mobile Documents/com~apple~CloudDocs/Documents/7mo Semestre/Mineria de datos/Ejemplos/Datasets/combined_data.csv"
)

# Análisis exploratorio de datos
print(df.describe())
print(df.info())
print(df.value_counts("label"))
"""
Columns
    -label
        '1' indicates that the email is classified as spam. count=43910
        '0' denotes that the email is legitimate (ham).     count=39538
    -text
        This column contains the actual content of the email messages

No hay un desbalanceo de clases
"""

# División de los datos
x, y = (df["text"], df["label"])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Vectorización
vectorizer = CountVectorizer()
x_train_transformed = vectorizer.fit_transform(x_train)

# Modelo de Naive Bayes
clf = MultinomialNB()
clf.fit(x_train_transformed, y_train)
x_test_transformed = vectorizer.transform(x_test)
y_pred = clf.predict(x_test_transformed)

# Arbol de decisión
clf_tree = DecisionTreeClassifier()
clf_tree.fit(x_train_transformed, y_train)
y_pred_tree = clf_tree.predict(x_test_transformed)

# modelo SVM
clf_svm = svm.SVC()
clf_svm.fit(x_train_transformed, y_train)
y_pred_svm = clf_svm.predict(x_test_transformed)

# Evaluación de los modelos
nb_accuracy = metrics.accuracy_score(y_test, y_pred)
nb_precision = metrics.precision_score(y_test, y_pred)
nb_recall = metrics.recall_score(y_test, y_pred)
nb_error_rate = 1 - metrics.accuracy_score(y_test, y_pred)
nb_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
nb_tn, nb_fp, nb_fn, nb_tp = metrics.confusion_matrix(y_test, y_pred).ravel()
nb_specificity = nb_tn / (nb_tn + nb_fp)

tree_accuracy = metrics.accuracy_score(y_test, y_pred_tree)
tree_precision = metrics.precision_score(y_test, y_pred_tree)
tree_recall = metrics.recall_score(y_test, y_pred_tree)
tree_error_rate = 1 - metrics.accuracy_score(y_test, y_pred_tree)
tree_confusion_matrix = metrics.confusion_matrix(y_test, y_pred_tree)
tree_tn, tree_fp, tree_fn, tree_tp = metrics.confusion_matrix(
    y_test, y_pred_tree
).ravel()
tree_specificity = tree_tn / (tree_tn + tree_fp)

svm_accuracy = metrics.accuracy_score(y_test, y_pred_svm)
svm_precision = metrics.precision_score(y_test, y_pred_svm)
svm_recall = metrics.recall_score(y_test, y_pred_svm)
svm_error_rate = 1 - metrics.accuracy_score(y_test, y_pred_svm)
svm_confusion_matrix = metrics.confusion_matrix(y_test, y_pred_svm)
svm_tn, svm_fp, svm_fn, svm_tp = metrics.confusion_matrix(y_test, y_pred_svm).ravel()
svm_specificity = svm_tn / (svm_tn + svm_fp)

print("---------------------------- Naive Bayes ----------------------------")
print("Accuracy:", nb_accuracy)
print("precision:", nb_precision)
print("recall:", nb_recall)
print("Error rate:", nb_error_rate)
print("Confusion Matrix:\n", nb_confusion_matrix)
print("Specificity:", nb_specificity)
print("\n")

print("--------------------------- Decision Tree ---------------------------")
print("Accuracy:", tree_accuracy)
print("precision:", tree_precision)
print("recall:", tree_recall)
print("Error rate:", tree_error_rate)
print("Confusion Matrix:\n", tree_confusion_matrix)
print("Specificity:", tree_specificity)
print("\n")

print("-------------------------------- SVM --------------------------------")
print("Accuracy:", svm_accuracy)
print("precision:", svm_precision)
print("recall:", svm_recall)
print("Error rate:", svm_error_rate)
print("Confusion Matrix:\n", svm_confusion_matrix)
print("Specificity:", svm_specificity)
print("\n")

metrics_data = {
    "Naive Bayes": [
        nb_accuracy,
        nb_precision,
        nb_recall,
        nb_error_rate,
        nb_specificity,
    ],
    "Decision Tree": [
        tree_accuracy,
        tree_precision,
        tree_recall,
        tree_error_rate,
        tree_specificity,
    ],
    "SVM": [svm_accuracy, svm_precision, svm_recall, svm_error_rate, svm_specificity],
}

df_metrics = pd.DataFrame(
    metrics_data, index=["Accuracy", "Precision", "Recall", "Error Rate", "Specificity"]
)

df_metrics.plot(kind="bar", figsize=(10, 10))
plt.ylabel("Score")
plt.title("Comparación de modelos")
plt.show()
