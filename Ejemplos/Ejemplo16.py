import tensorflow  as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

iris = datasets.load_iris()
x = iris.data
y = iris.target

encoder = OneHotEncoder(sparse=False) # Nos ayuda a categorizar
y = encoder.fit_transform(y.reshape(-1, 1))
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#contruir el modelo ANN
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=8, activation='relu', input_shape=(4,)), #capa oculta
    tf.keras.layers.Dense(units=3, activation='softmax'), #capa de salida con 3 neuronas
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50)
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')