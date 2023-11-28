import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "/Users/danny/Library/Mobile Documents/com~apple~CloudDocs/Documents/7mo Semestre/Mineria de datos/Ejemplos/Datasets/healthcare_dataset.csv"
)

"""
Información del Dataset
    print(df.describe())
    print(df.info())
- Name: This column represents the name of the patient associated with the healthcare record.
- Age: The age of the patient at the time of admission, expressed in years.
- Gender: Indicates the gender of the patient, either "Male" or "Female."
- Blood Type: The patient's blood type, which can be one of the common blood types (e.g., "A+", "O-", etc.).
- Medical Condition: This column specifies the primary medical condition or diagnosis associated with the patient, such as "Diabetes," "Hypertension," "Asthma," and more.
- Date of Admission: The date on which the patient was admitted to the healthcare facility.
- Doctor: The name of the doctor responsible for the patient's care during their admission.
- Hospital: Identifies the healthcare facility or hospital where the patient was admitted.
- Insurance Provider: This column indicates the patient's insurance provider, which can be one of several options, including "Aetna," "Blue Cross," "Cigna," "UnitedHealthcare," and "Medicare."
- Billing Amount: The amount of money billed for the patient's healthcare services during their admission. This is expressed as a floating-point number.
- Room Number: The room number where the patient was accommodated during their admission.
- Admission Type: Specifies the type of admission, which can be "Emergency," "Elective," or "Urgent," reflecting the circumstances of the admission.
- Discharge Date: The date on which the patient was discharged from the healthcare facility, based on the admission date and a random number of days within a realistic range.
- Medication: Identifies a medication prescribed or administered to the patient during their admission. Examples include "Aspirin," "Ibuprofen," "Penicillin," "Paracetamol," and "Lipitor."
- Test Results: Describes the results of a medical test conducted during the patient's admission. Possible values include "Normal," "Abnormal," or "Inconclusive," indicating the outcome of the test.
"""
# Factorización de los datos
df["Gender"] = pd.factorize(df["Gender"])[0]
df["Blood Type"] = pd.factorize(df["Blood Type"])[0]
df["Medical Condition"] = pd.factorize(df["Medical Condition"])[0]
df["Insurance Provider"] = pd.factorize(df["Insurance Provider"])[0]
df["Admission Type"] = pd.factorize(df["Admission Type"])[0]
df["Medication"] = pd.factorize(df["Medication"])[0]
df["Test Results"] = pd.factorize(df["Test Results"])[0]

# Correlación entre variables
dfCorr = pd.DataFrame(
    df,
    columns=[
        "Age",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Billing Amount",
        "Admission Type",
        "Medication",
        "Test Results",
    ],
)
correlation_matrix = dfCorr.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True)
plt.show()
# Observación de los resultados de la correlación: valores bajos para todas las variables consideradas!

# Seleccion de Caracteristicas
df_features = df[
    [
        "Age",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Billing Amount",
        "Admission Type",
        "Medication",
    ]
]
df_target = df["Test Results"]

# Normalización de los datos
scaler = StandardScaler()
df_features_standardized = scaler.fit_transform(df_features)
df_features_standardized = pd.DataFrame(
    df_features_standardized, columns=df_features.columns
)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    df_features_standardized, df_target, test_size=0.2, random_state=42
)

# Naive Bayes
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)

# Evaluación de los modelos
# Naive Bayes
nb_accuracy = metrics.accuracy_score(y_test, y_pred)
nb_precision = metrics.precision_score(y_test, y_pred, average="macro")
nb_recall = metrics.recall_score(y_test, y_pred, average="macro")
nb_error_rate = 1 - metrics.accuracy_score(y_test, y_pred)
nb_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
nb_total = np.sum(nb_confusion_matrix)

# Random Forest
rf_accuracy = metrics.accuracy_score(y_test, y_pred_rf)
rf_precision = metrics.precision_score(y_test, y_pred_rf, average="macro")
rf_recall = metrics.recall_score(y_test, y_pred_rf, average="macro")
rf_error_rate = 1 - metrics.accuracy_score(y_test, y_pred_rf)
rf_confusion_matrix = metrics.confusion_matrix(y_test, y_pred_rf)
rf_total = np.sum(rf_confusion_matrix)

print("\n---------------Naive Bayes---------------")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("Error Rate:", nb_error_rate)
print("Confusion Matrix:\n", nb_confusion_matrix)
for i in range(3):
    TN = (
        nb_total
        - np.sum(nb_confusion_matrix[i, :])
        - np.sum(nb_confusion_matrix[:, i])
        + nb_confusion_matrix[i, i]
    )
    FP = np.sum(nb_confusion_matrix[:, i]) - nb_confusion_matrix[i, i]
    specificity = TN / (TN + FP)
    print(f"Specificity for class {i}: {specificity}")
print("\n")

print("\n---------------Random Forest---------------")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("Error Rate:", rf_error_rate)
print("Confusion Matrix:\n", rf_confusion_matrix)
for i in range(3):
    TN = (
        nb_total
        - np.sum(rf_confusion_matrix[i, :])
        - np.sum(rf_confusion_matrix[:, i])
        + rf_confusion_matrix[i, i]
    )
    FP = np.sum(rf_confusion_matrix[:, i]) - rf_confusion_matrix[i, i]
    specificity = TN / (TN + FP)
    print(f"Specificity for class {i}: {specificity}")
print("\n")

metrics_data = {
    'Naive Bayes': [nb_accuracy, nb_precision, nb_recall, nb_error_rate],
    'Random Forest': [rf_accuracy, rf_precision, rf_recall, rf_error_rate]
}

df_metrics = pd.DataFrame(metrics_data, index=['Accuracy', 'Precision', 'Recall', 'Error Rate'])

df_metrics.plot(kind='bar', figsize=(10, 10))
plt.ylabel('Score')
plt.title('Comparación de modelos')
plt.show()