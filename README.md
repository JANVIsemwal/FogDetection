# 🌫️ Fog Detection AI (Authority Verification System)

This repository contains the backend Machine Learning component for an Authority Environmental Verification App. It uses a highly optimized Neural Network to look at a user-uploaded photograph and instantly determine if the environment is **FOGGY** or **CLEAR**.

## 🚀 Features
* **High Accuracy**: Achieves ~98.7% accuracy on environmental image classification.
* **Mobile-Ready AI**: Uses Google's `MobileNetV2` architecture, which is specifically designed for extremely fast, low-power execution on edge devices and smartphones.
* **Visual Inference**: Includes a visual GUI verification script for testing completely unseen data in a Matplotlib dashboard.

---

## 📁 Project Structure

* **`train_fog.py`**: The main training pipeline. It loads the dataset, splits it into 80% Training and 20% Validation, handles Data Augmentation, and trains the MobileNetV2 classifier.
* **`test_fog.py`**: A quick console-based utility script to test a single image file against the model.
* **`predict_unseen_fog.py`**: A visual GUI script that iterates through a folder of unseen images, predicts their class (FOG DETECTED or CLEAR), and displays the photos with their AI-confidence scores visibly overlaid.
* **`fog_detection_model.keras`**: *(Not included in Git)* The trained, saved Artificial Brain.
* **`FogDataset/`**: *(Not included in Git)* The training dataset directory.
* **`Unseenfog/`**: *(Not included in Git)* Directory for putting scraped testing images.

---

## 🛠️ How to Run Locally

### 1. Setup Your Environment
Ensure you have Python installed along with TensorFlow and Pillow.
```bash
pip install tensorflow pillow matplotlib numpy
```

### 2. Prepare the Data
Because datasets are massive, they are excluded from this repository. You must create the following folders:
- Create `FogDataset/clear/` and `FogDataset/foggy/` and place thousands of training images inside.
- Create `Unseenfog/` and place random test photos downloaded from the internet here.

### 3. Train the Brain
Run the training script to analyze all photos in the FogDataset and generate the `.keras` model file.
```bash
python train_fog.py
```

### 4. Test the Artificial Intelligence
To visually test the model's accuracy on the photos in your `Unseenfog/` folder, run the predictor. A window will pop up showing the images with red/green "FOG DETECTED" labels.
```bash
python predict_unseen_fog.py
```
