from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.backend import clear_session
import h5py
import matplotlib.pyplot as plt

clear_session()

# Read data from files
data_train = h5py.File("../data/processed/train_data.h5", "r")
data_val = h5py.File("../data/processed/val_data.h5", "r")
data_test = h5py.File("../data/processed/test_data.h5", "r")

X_train = data_train["X"]
y_train = data_train["y"]
X_val = data_val["X"]
y_val = data_val["y"]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

shape_input = X_train.shape[1:]
length_output = y_train.shape[1]

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=shape_input))
model.add(LSTM(64))
model.add(Dense(length_output))

print(model.summary())

model.compile(loss=MeanSquaredError(), optimizer=RMSprop())
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
print('Done!')