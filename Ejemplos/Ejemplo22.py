import numpy as np
import matplotlib.pyplot as plt

# Datos de ventas de helados y temperaturas
np.random.seed(0)
temperaturas = np.linspace(20, 40, 50)
ventas_helados = 50 + 2 * temperaturas + np.random.normal(0, 5, 50) 
#50 valor base + coeficiente de relacio + el dato radom para generar ruido

# Datos de horas de estudio y calificaciones
horas_estudio = np.linspace(1, 5, 50)
calificaciones = 100 - 10 * horas_estudio + np.random.normal(0, 5, 50)

# Datos sin correlacion
horas_suenio = np.linspace(4, 10, 50)
ventas_helados2 = 50 + np.random.normal(0, 10, 50)

# Calcular la correlacion de Pearson
no_correlation = np.corrcoef(horas_suenio, ventas_helados2)[0, 1]
correlation_pos = np.corrcoef(ventas_helados, temperaturas)[0, 1]
correlation_neg = np.corrcoef(horas_estudio, calificaciones)[0, 1]

plt.figure(figsize=(15, 5))

# grafico para la correlacion positiva
plt.subplot(1, 3, 1)
plt.scatter(temperaturas, ventas_helados, label=f'Correlación = {correlation_pos:.2f}')
plt.plot(temperaturas, temperaturas * correlation_pos, color='red', linestyle='--')
plt.xlabel('Temperaturas')
plt.ylabel('Ventas de Helados')
plt.legend()
plt.title('Correlación Positiva')
plt.grid(True)

# grafico para la correlacion negativa
plt.subplot(1, 3, 2)
plt.scatter(horas_estudio, calificaciones, label=f'Correlación = {correlation_neg:.2f}')
plt.plot(horas_estudio, horas_estudio * correlation_neg, color='red', linestyle='--')
plt.xlabel('Horas de Estudio')
plt.ylabel('Calificaciones')
plt.legend()
plt.title('Correlación Negativa')
plt.grid(True)

# grafico para la no correlacion
plt.subplot(1, 3, 3)
plt.scatter(horas_suenio, ventas_helados2, label=f'Correlación = {no_correlation:.2f}')
plt.plot(horas_suenio, ventas_helados2 * no_correlation, color='red', linestyle='--')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('No Correlación')
plt.grid(True)

plt.tight_layout()

plt.show()
