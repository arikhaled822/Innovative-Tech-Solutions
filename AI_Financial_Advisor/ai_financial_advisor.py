import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and preprocess the data
data = pd.read_csv('financial_data.csv')
X = data.drop('target', axis=1).values
y = data['target'].values

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Save the model
model.save('financial_advisor_model.h5')

# Usage example
def predict_advice(input_data):
    model = tf.keras.models.load_model('financial_advisor_model.h5')
    prediction = model.predict(input_data)
    return prediction
