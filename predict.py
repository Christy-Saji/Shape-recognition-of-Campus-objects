import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sys

IMG_SIZE = 224
MODEL_PATH = "models/best_model.h5"

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (must match folder order)
class_names = ['circle', 'rectangle', 'triangle']

# ======================
# LOAD IMAGE
# ======================
img_path = sys.argv[1]

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ======================
# PREDICT
# ======================
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

print(f"\nPredicted Shape: {predicted_class}")
print("Confidence:", np.max(prediction)*100, "%")