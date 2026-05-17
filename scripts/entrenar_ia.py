import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Carga el archivo DATASET generado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_data = os.path.join(BASE_DIR, '..', 'data', 'datos_moda.csv')
ruta_scaler = os.path.join(BASE_DIR, '..', 'models', 'scaler_moda.joblib')
ruta_modelo = os.path.join(BASE_DIR, '..', 'models', 'modelo_moda.keras')
data = pd.read_csv(ruta_data)

# Separamos entradas (X) de las etiquetas (y)
X = data[['estatura_cm', 'peso_kg']]
y = data['talla_label']

# Dividimos en 80% para entrenar y 20% para examen final PRIMERO
# (Para evitar que el scaler aprenda de los datos de test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamos los datos ajustando SOLO en los de entrenamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardamos el escalador para usarlo en el recomendador sin volver a entrenarlo
os.makedirs(os.path.dirname(ruta_scaler), exist_ok=True)
joblib.dump(scaler, ruta_scaler)
print(f"Escalador guardado en: {ruta_scaler}")

# 3. Definir la Arquitectura de la Red Neuronal
model = keras.Sequential([
    # Definición de la forma de entrada (2 variables: peso y estatura)
    keras.Input(shape=(2,)),
    # Capa de neuronas inicial
    keras.layers.Dense(16, activation='relu'),
    # Dropout para evitar el sobreajuste (apaga 20% de neuronas aleatoriamente)
    keras.layers.Dropout(0.2),
    # Capa oculta para patrones complejos
    keras.layers.Dense(12, activation='relu'),
    # Capa de salida: 6 neuronas (S, M, L, XL, XXL, XXXL)
    keras.layers.Dense(6, activation='softmax')
])

# 4. Compilación
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early Stopping para detener el entrenamiento cuando deje de mejorar
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 5. Entrenamiento (El momento del aprendizaje)
print("Estudiando datos...")
history = model.fit(
    X_train_scaled, y_train, 
    epochs=100, 
    validation_split=0.2, 
    callbacks=[early_stop],
    verbose=1
)

# 6. Evaluación
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nEntrenamiento completado. Precisión en datos de prueba: {accuracy*100:.2f}%")

# 7. Guardar el modelo en formato .keras (moderno)
os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
model.save(ruta_modelo)
print(f"Modelo guardado en: {ruta_modelo}")