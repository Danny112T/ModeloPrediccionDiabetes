import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, svm
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option("display.max_rows", None)

# Carga de los datasets
df0 = pd.read_csv(
    "/Users/danny/Library/Mobile Documents/com~apple~CloudDocs/Documents/7mo Semestre/Mineria de datos/Ejemplos/ProyectoFinal/Datasets/ensadul2022_entrega_w.csv",
    delimiter=";",
    low_memory=False,
)
df1 = pd.read_csv(
    "/Users/danny/Library/Mobile Documents/com~apple~CloudDocs/Documents/7mo Semestre/Mineria de datos/Ejemplos/ProyectoFinal/Datasets/ensaantro2022_entrega_w.csv",
    delimiter=";",
    low_memory=False,
)
df2 = pd.read_csv(
    "/Users/danny/Library/Mobile Documents/com~apple~CloudDocs/Documents/7mo Semestre/Mineria de datos/Ejemplos/ProyectoFinal/Datasets/Determinaciones_bioquímicas_cronicas_deficiencias_9feb23.csv",
    delimiter=";",
    low_memory=False,
)
df3 = pd.read_csv(
    "/Users/danny/Library/Mobile Documents/com~apple~CloudDocs/Documents/7mo Semestre/Mineria de datos/Ejemplos/ProyectoFinal/Datasets/ensafisica2022_adultos_entrega_w.csv",
    delimiter=";",
    low_memory=False,
)

# Selección de columnas de interés
df0_selected_columns = ["FOLIO_INT", "a0301"]

df1_selected_columns = [
    "FOLIO_INT",
    "an01_1",
    "an01_2",
    "an04_1",
    "an08_2",
    "an12_1",
    "an15_1",
    "an17_1",
    "an21_1",
]

df2_selected_columns = [
    "FOLIO_INT",
    "h0302",
    "h0303",
    "valor_HB1AC",
    "sc",
    "hb02",
    "valor_AC_URICO",
    "valor_CREAT",
    "valor_GLU_SUERO",
    "valor_INSULINA",
    "valor_TRIG",
    "valor_EAG",
    "valor_VIT_B12",
]

df3_selected_columns = [
    "FOLIO_INT",
    "fa0400",
    "fa0401",
    "fa0403",
    "fa0405",
    "fa0407h",
    "fa0408",
    "fa0409h",
    "fa0409m",
    "fa0410",
    "fa0411",
    "fa0412",
    "fa0413",
    "fa0414",
    "fa0415",
]

# Union de los datasets por el campo FOLIO_INT
merged_df = pd.merge(
    df0[df0_selected_columns], df1[df1_selected_columns], on="FOLIO_INT", how="inner"
)
merged_df = pd.merge(merged_df, df2[df2_selected_columns], on="FOLIO_INT", how="inner")
merged_df = pd.merge(merged_df, df3[df3_selected_columns], on="FOLIO_INT", how="inner")

merged_df.drop(merged_df[merged_df["a0301"] == 2].index, inplace=True)

columns_to_convert = [
    "hb02",
    "valor_AC_URICO",
    "valor_CREAT",
    "valor_GLU_SUERO",
    "valor_INSULINA",
    "valor_TRIG",
    "valor_EAG",
    "valor_HB1AC",
    "valor_VIT_B12",
    "an01_1",
    "an01_2",
    "an04_1",
    "an08_2",
    "an12_1",
    "an15_1",
    "an17_1",
    "an21_1",
]

merged_df[columns_to_convert] = merged_df[columns_to_convert].apply(
    pd.to_numeric, errors="coerce"
)

# limpieza de datos
merged_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
null_count = merged_df.isnull().sum()
print(null_count)

merged_df.to_csv(
    "/Users/danny/Library/Mobile Documents/com~apple~CloudDocs/Documents/7mo Semestre/Mineria de datos/Ejemplos/ProyectoFinal/Datasets/Pd.csv",
    index=False,
)

# Cambiar datos de tipo string a numericos
merged_df[columns_to_convert] = merged_df[columns_to_convert].astype(float)

# Llenar valores nulos con knn e imputacion columna por columna
imputer = KNNImputer(n_neighbors=5)

for column in merged_df.columns:
    column_values = merged_df[[column]]
    imputed_column_values = imputer.fit_transform(column_values).ravel()
    merged_df[column] = imputed_column_values


null_count2 = merged_df.isnull().sum()
print(null_count2)

correlation_matrix = merged_df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True)
#plt.show()

# normalizacion de datos
