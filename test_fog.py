import tensorflow as tf
import numpy as np

# 1. Load our trained Brain
print("Loading our trained brain...")
model = tf.keras.models.load_model('fog_detection_model.keras')

# 2. Define the exact path to an image we want to test.
# Let's test it on one of our Clear images to see if it gets it right!
IMAGE_PATH = r"FogDataset\foggy\Fog1 (100).jpg" 

print(f"Testing image: {IMAGE_PATH}")

# 3. Load the image and resize it to 224x224 (what the model expects)
img = tf.keras.utils.load_img(IMAGE_PATH, target_size=(224, 224))

# 4. Neural Networks expect a "batch" of images, not just one. 
# We use img_to_array and expand_dims to fake a batch of size 1.
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create our batch of 1

# 5. Make the Prediction!
prediction = model.predict(img_array)

# 6. Interpret the Result
score = prediction[0][0] # The raw score between 0.0 and 1.0

print("-" * 30)
if score > 0.5:
    print(f"Result: The model thinks this is FOGGY! (Confidence: {score*100:.2f}%)")
else:
    # If the score is close to 0, confidence of it being clear is 1 - score
    print(f"Result: The model thinks this is CLEAR! (Confidence: {(1-score)*100:.2f}%)")
print("-" * 30)
