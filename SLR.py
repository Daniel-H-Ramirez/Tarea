import numpy as np

class SLR:
    def __init__(self):
        # Datos del constructor
        self.data = {
            'Advertising': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'Sales': [2, 4, 6, 8, 10, 12, 14, 16, 18]
            
        }
        self.beta_0 = None
        self.beta_1 = None

    def fit(self):
        # Convertir X e y a arrays de NumPy
        X = np.array(self.data['Advertising'])
        y = np.array(self.data['Sales'])
        n = len(X)
        
        # Calcular las sumas necesarias 
        sum_x = np.sum(X)
        sum_y = np.sum(y)
        sum_xy = np.sum(X * y)
        sum_x_squared = np.sum(X ** 2)
        
        # Calcular beta_0 y beta_1
        denominator = n * sum_x_squared - sum_x ** 2
        self.beta_0 = (sum_x_squared * sum_y - sum_x * sum_xy) / denominator
        self.beta_1 = (n * sum_xy - sum_x * sum_y) / denominator

    def predict(self, X):
        if self.beta_0 is None or self.beta_1 is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        return self.beta_0 + self.beta_1 * X

    def get_equation(self):
        # Devuelve un string con el resultado
        return f"ŷ = {self.beta_0:.4f} + {self.beta_1:.4f} * x"

def main():
    # Crear y entrenar el modelo
    model = SLR()
    model.fit()
    
    # Imprimir la regresión
    equation = model.get_equation()
    print("Ecuación de Regresión:", equation)
    
    # Ingresar valores a elegir de publicidad
    while True:
        try:
            user_input = input("Ingrese un valor de Publicidad para predecir Ventas (o 'salir' para terminar): ")
            if user_input.lower() == 'salir':
                break
            x_value = float(user_input)
            prediction = model.predict(x_value)
            print(f"Predicción de Ventas para Publicidad = {x_value}: {prediction:.4f}")
        except ValueError:
            print("Por favor, ingrese un número válido o 'salir' para terminar.")

if __name__ == "__main__":
    main()
