import tensorflow as tf
import os
# We started with 2,335 total images (1,128 Clear and 1,207 Foggy).
# 1. Define where our data lives and our target image parameters
DATASET_DIR = "FogDataset"
BATCH_SIZE = 32         # The model will look at 32 images at a time before updating its memory
IMG_HEIGHT = 224        # Standard height 
IMG_WIDTH = 224         # Standard width

print("Loading training data...")

# 2. Load the Training split (80% of the data)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,    # Reserve 20% of data for validation
    subset="training",       # Tell Keras this is the training batch
    seed=123,                # Random seed ensures our 80/20 split is consistent every time we run
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

print("Loading validation data...")

# 3. Load the Validation split (20% of the data)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",     # Tell Keras this is the hidden validation batch
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# 4. Print out the class names to confirm it found our two folders
class_names = train_ds.class_names
print(f"Discovered classes: {class_names}")

# 5. Optimize dataset performance (crucial for fast training)
# 'cache()' keeps images in RAM instead of reading from disk every time.
# 'prefetch()' prepares the next batch of 32 images while the current batch is training.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- STEP 2: BUILDING THE MODEL ---
print("Building the Neural Network...")

# 1. Load the pre-trained MobileNetV2 (without its original classification brain)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False, # We don't want the 1000-class output, we want our own Custom 2-class output!
    weights='imagenet' # Pre-trained on 1.4 million images
)

# --- STEP 2: BUILDING THE MODEL (WITH AUGMENTATION) ---
print("Building the Neural Network...")

# 1. Data Augmentation Layers (The game changer)
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
])

# 2. Load the pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False, 
    weights='imagenet' 
)

# 3. Freeze the base model for the first round of training
base_model.trainable = False

# 4. Build our Custom "Head" 
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

x = data_augmentation(inputs) # Apply our random flips/rotations to the input
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) 
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)  
x = tf.keras.layers.Dropout(0.2)(x)             
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  

model = tf.keras.Model(inputs, outputs)

# 5. Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# --- STEP 3: INITIAL TRAINING ---
print("Starting Initial Training (5 epochs)...")
history = model.fit(train_ds, epochs=5, validation_data=val_ds)

# --- STEP 4: FINE-TUNING (THE SECRET WEAPON) ---
print("\nUnfreezing the top layers for Fine-Tuning...")

# Unfreeze the base model
base_model.trainable = True

# Let's see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards (leaving the bottom ~100 layers frozen)
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Because we are making changes to the core brain, we MUST use a much smaller learning rate 
# (10x smaller) so we don't accidentally destroy its knowledge.
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001), 
              metrics=['accuracy'])

print("Starting Fine-Tuning Training (5 extra epochs)...")
total_epochs = 5 + 5 # 5 initial + 5 fine-tuning

history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1], # Start from epoch 5
                         validation_data=val_ds)

# --- STEP 5: SAVE THE BRAIN ---
model.save("fog_detection_model.keras")
print("Training & Fine-Tuning complete! Model successfully saved as 'fog_detection_model.keras'!")
