import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.api.preprocessing.image import  load_img, img_to_array

csv_path = "/Users/lucasbarszcz/Downloads/Plank_detection.v1i.tensorflow/train/_annotations.csv"
df = pd.read_csv(csv_path)

image_dir = "/Users/lucasbarszcz/Downloads/Plank_detection.v1i.tensorflow/train/"

def load_and_preprocess_image(filename, target_size=(128, 128)):
    img_path = image_dir + filename
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array
images = np.array([load_and_preprocess_image(filename) for filename in df['filename']])
labels = df['class'].values

# 4. Codificar as classes (se forem categóricas)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
# 5. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# 6. Criar o modelo CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Número de classes
])

# 7. Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 8. Treinar o modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 9. Avaliar o modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {test_accuracy}")

# 10. Salvar o modelo (opcional)
model.save("meu_modelo.h5")