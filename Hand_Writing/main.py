import os
import random
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import time

data_folder = "Letters"
num_classes = 26


def load_dataset(folder):
    images = []
    labels = []

    for file_name in os.listdir(folder):
        if file_name.endswith(".png"):
            img_path = os.path.join(folder, file_name)
            label = file_name.split("_")[0].lower()  # Harfi al, küçük harfe dönüştür
            images.append(img_path)
            labels.append(label)

    if not images:
        raise ValueError("No valid images found in the specified folder.")

    return pd.DataFrame({"Image": images, "Label": labels})


dataset = load_dataset(data_folder)
def preprocess_image(image_path, target_size=(100, 100)):
    img = Image.open(image_path).convert("L")  # Gri tonlamalı resim

    # Resmi hedef boyuta getir
    img = img.resize(target_size[::-1])

    # Resmi normalize et
    img_array = np.array(img) / 255.0

    # Resmi modele uygun formata getir
    img_array = np.expand_dims(img_array, axis=-1)  # Gri tonlamalı olduğu için kanal sayısını ekleyin
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutunu ekleyin

    return img_array

print("Veri Seti Örnekleri:")
print(dataset.head())


def preprocess_data(dataset, target_size=(100, 100)):
    if not dataset["Image"].any():
        raise ValueError("No valid images in the dataset.")
    X = []
    y = []
    for index, row in dataset.iterrows():
        img_path = row["Image"]
        label = row["Label"]
        # Resmi yükle, normalize et, ve boyutunu ayarla
        img = Image.open(img_path).convert("L")  # L: Gri tonlamalı resim
        img = img.resize(target_size)
        img = np.array(img) / 255.0
        X.append(img)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y


try:
    X, y = preprocess_data(dataset)
except ValueError as e:
    print(f"Error during preprocessing: {e}")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_cnn = models.Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout ekleyin
    layers.Dense(num_classes, activation='softmax')
])

model_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Model 1 (CNN)
start_time = time.time()
model_cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
end_time = time.time()
cnn_execution_time = end_time - start_time
# Modelin özetini al
model_cnn.summary()

# İlk Conv2D katmanındaki ağırlıkları al
conv1_weights = model_cnn.layers[0].get_weights()[0]

# İlk filtreyi seçin (3. boyut)
filter_index = 0
conv1_filter = conv1_weights[:, :, 0, filter_index]

# Ağırlıkları ısı haritası olarak görselleştir
plt.figure(figsize=(10, 8))
sns.heatmap(conv1_filter, cmap='viridis', annot=True, fmt=".2f")
plt.title(f'Conv2D Layer 1, Filter {filter_index + 1} Weights')
plt.show()

model_lstm = models.Sequential([
    layers.LSTM(64, input_shape=(100, 100)),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_lstm.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Model 2 (LSTM)
start_time = time.time()
model_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
end_time = time.time()
lstm_execution_time = end_time - start_time

model_bilstm = models.Sequential([
    layers.Bidirectional(layers.LSTM(64), input_shape=(100, 100)),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_bilstm.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# Model 3 (BiLSTM)
start_time = time.time()
model_bilstm.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
end_time = time.time()
bilstm_execution_time = end_time - start_time

model_stacked_gru = models.Sequential([
    layers.GRU(32, return_sequences=True, input_shape=(100, 100)),
    layers.GRU(32),
    layers.Dense(num_classes, activation='softmax')
])

model_stacked_gru.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# Model 4 (Stacked GRU)
start_time = time.time()
model_stacked_gru.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
end_time = time.time()
stacked_gru_execution_time = end_time - start_time

model_dense = models.Sequential([
    layers.Flatten(input_shape=(100, 100)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model_dense.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Model 5 (Dense)
start_time = time.time()
model_dense.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
end_time = time.time()
dense_execution_time = end_time - start_time

model_simple = models.Sequential([
    layers.Flatten(input_shape=(100, 100)),
    layers.Dense(num_classes, activation='softmax')
])

model_simple.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# Model 6 (Simple)
start_time = time.time()
model_simple.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
end_time = time.time()
simple_execution_time = end_time - start_time

# Display execution times
print("\nExecution Times:")
print(f"Model 1 (CNN): {cnn_execution_time} seconds")
print(f"Model 2 (LSTM): {lstm_execution_time} seconds")
print(f"Model 3 (BiLSTM): {bilstm_execution_time} seconds")
print(f"Model 4 (Stacked GRU): {stacked_gru_execution_time} seconds")
print(f"Model 5 (Dense): {dense_execution_time} seconds")
print(f"Model 6 (Simple): {simple_execution_time} seconds")
# Sort and display models based on execution time
execution_times = {
    "Model 1 (CNN)": cnn_execution_time,
    "Model 2 (LSTM)": lstm_execution_time,
    "Model 3 (BiLSTM)": bilstm_execution_time,
    "Model 4 (Stacked GRU)": stacked_gru_execution_time,
    "Model 5 (Dense)": dense_execution_time,
    "Model 6 (Simple)": simple_execution_time,
}

sorted_models = sorted(execution_times.items(), key=lambda x: x[1])
print("\nModels sorted by execution time:")
for model, time in sorted_models:
    print(f"{model}: {time} seconds")
print("\nDoğruluk Değerleri:")
print(f"Model 1 (CNN): {model_cnn.evaluate(X_test, y_test)[1]}")
print(f"Model 2 (LSTM): {model_lstm.evaluate(X_test, y_test)[1]}")
print(f"Model 3 (BiLSTM): {model_bilstm.evaluate(X_test, y_test)[1]}")
print(f"Model 4 (Stacked GRU): {model_stacked_gru.evaluate(X_test, y_test)[1]}")
print(f"Model 5 (Dense): {model_dense.evaluate(X_test, y_test)[1]}")
print(f"Model 6 (Simple): {model_simple.evaluate(X_test, y_test)[1]}")




label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

etiket_harf_sozlugu = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}
while True:
    user_input = input("Bir Harf Giriniz :  ")
    random_number = random.randint(0, 9)
    image_path = f"Letters/{user_input}_{random_number}.png"
    input_image = preprocess_image(image_path)

    img = Image.open(image_path)
    img.show()

    # Model 1 (CNN) Tahmini
    predictions_cnn = model_cnn.predict(input_image)
    predicted_label_cnn = label_encoder.inverse_transform(np.argmax(predictions_cnn, axis=1))[0]
    predicted_harf_cnn = etiket_harf_sozlugu[predicted_label_cnn]

    # Model 2 (LSTM) Tahmini
    predictions_lstm = model_lstm.predict(input_image)
    predicted_label_lstm = label_encoder.inverse_transform(np.argmax(predictions_lstm, axis=1))[0]
    predicted_harf_lstm = etiket_harf_sozlugu[predicted_label_lstm]

    # Model 3 (BiLSTM) Tahmini
    predictions_bilstm = model_bilstm.predict(input_image)
    predicted_label_bilstm = label_encoder.inverse_transform(np.argmax(predictions_bilstm, axis=1))[0]
    predicted_harf_bilstm = etiket_harf_sozlugu[predicted_label_bilstm]

    # Model 4 (Stacked GRU) Tahmini
    predictions_stacked_gru = model_stacked_gru.predict(input_image)
    predicted_label_stacked_gru = label_encoder.inverse_transform(np.argmax(predictions_stacked_gru, axis=1))[0]
    predicted_harf_stacked_gru = etiket_harf_sozlugu[predicted_label_stacked_gru]

    # Model 5 (Dense) Tahmini
    predictions_dense = model_dense.predict(input_image)
    predicted_label_dense = label_encoder.inverse_transform(np.argmax(predictions_dense, axis=1))[0]
    predicted_harf_dense = etiket_harf_sozlugu[predicted_label_dense]

    # Model 6 (Simple) Tahmini
    predictions_simple = model_simple.predict(input_image)
    predicted_label_simple = label_encoder.inverse_transform(np.argmax(predictions_simple, axis=1))[0]
    predicted_harf_simple = etiket_harf_sozlugu[predicted_label_simple]

    # Tahminleri yazdır
    print("Model 1 (CNN) Tahmini:", predicted_harf_cnn)
    print("Model 2 (LSTM) Tahmini:", predicted_harf_lstm)
    print("Model 3 (BiLSTM) Tahmini:", predicted_harf_bilstm)
    print("Model 4 (Stacked GRU) Tahmini:", predicted_harf_stacked_gru)
    print("Model 5 (Dense) Tahmini:", predicted_harf_dense)
    print("Model 6 (Simple) Tahmini:", predicted_harf_simple)
