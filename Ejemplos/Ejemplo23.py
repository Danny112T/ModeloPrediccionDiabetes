import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
	# Numero de observacione s
	n = np.size(x)

	# Vectores (x)  (y)
	m_x = np.mean(x)
	m_y = np.mean(y)

	# Calcular la desviación cruzada y la desviación sobre x
	SS_xy = np.sum(y*x) - n*m_y*m_x
	SS_xx = np.sum(x*x) - n*m_x*m_x

	# Calcular coeficientes de regresión
	b_1 = SS_xy / SS_xx
	b_0 = m_y - b_1*m_x
	return (b_0, b_1)

def plot_regression_line(x, y, b):
	# Trazar los puntos reales como diagrama de dispersión
	plt.scatter(x, y, color = "m",
			marker = "o", s = 30)

	# Vector de respuesta previsto
	y_pred = b[0] + b[1]*x

	# Trazar la línea de regresión
	plt.plot(x, y_pred, color = "g")
	plt.xlabel('x')
	plt.ylabel('y')

	# Mostrar la grafica
	plt.show()

def main():
	#Datos
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

	# Estimacion de los coeficientes
	b = estimate_coef(x, y)
	print("Estimacion de los coeficientes:\nb_0 = {} \
     \nb_1 = {}".format(b[0], b[1]))

	# trazar la línea de regresión
	plot_regression_line(x, y, b)

if __name__ == "__main__":
	main()
