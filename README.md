
# 📘 **MNIST Handwritten Digit Classification – CNN From Scratch**

This project implements a **Convolutional Neural Network (CNN)** from scratch (without pre-trained models) to classify handwritten digits from the **MNIST dataset**.
The entire workflow includes data preprocessing, model building, training, evaluation, and visualization.

---

## 📂 **Project Overview**

* **Dataset:** MNIST (70,000 grayscale images of digits 0–9)
* **Model:** Custom CNN built using **TensorFlow/Keras**
* **Task:** Multi-class classification
* **Input Size:** 28×28×1
* **Output:** Class probabilities for digits 0–9
* **Performance:** Achieved ~**98.9% test accuracy**

---

## 📊 **1. Dataset Preparation**

MNIST dataset is loaded using:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

### **Preprocessing Steps**

* Normalize pixel values → **0 to 1**
* Reshape images to **(28, 28, 1)** for CNN input

---

## 🧠 **2. CNN Model Architecture**

A custom CNN was created using TensorFlow's Sequential API.

### **Model Layers**

1. **Conv2D (8 filters, 5×5, ReLU)**
2. **MaxPooling2D (2×2)**
3. **Conv2D (16 filters, 5×5, ReLU)**
4. **MaxPooling2D (2×2)**
5. **Flatten**
6. **Dense (128 units, ReLU)**
7. **Dropout (0.2)**
8. **Dense (10 units, Softmax)**

### **Model Summary**

* **Total parameters:** ~37,610
* **Trainable:** 100%

The architecture is optimized for small grayscale images.

---

## ⚙️ **3. Model Compilation**

The model uses:

| Component     | Choice                        |
| ------------- | ----------------------------- |
| Optimizer     | Adam (lr = 0.001)             |
| Loss Function | SparseCategoricalCrossentropy |
| Metrics       | Accuracy                      |

---

## 🏋️ **4. Training the Model**

* **Epochs:** 10
* **Batch Size:** 128
* **Validation Split:** 20%

Training produced:

* Increasing accuracy
* Decreasing loss
* Minimal overfitting

---

## 📈 **5. Training Curves**

Two graphs are plotted:

* **Training vs Validation Loss**
* **Training vs Validation Accuracy**

These help visualize learning performance over epochs.

---

## 🧪 **6. Model Evaluation**

The model was evaluated on the test dataset:

```
Test accuracy: ~98.9%
Test loss: ~0.0379
```

Very high performance and generalization.

---

## 🔍 **7. Predictions & Visualization**

The notebook displays a **5×5 grid** of test images with:

* Predicted labels
* True labels

This helps visually inspect correctness and failure cases.

---

## 🎯 **8. Key Features of This Project**

✔ Fully custom CNN (no pre-trained models)
✔ Clean training workflow
✔ Visualization of images, training curves, and predictions
✔ High accuracy on MNIST
✔ Beginner-friendly deep learning project

---

## 📦 **Technologies Used**

* TensorFlow / Keras
* NumPy
* Matplotlib
* Python 3.x
* Google Colab

---

## 🚀 **How to Run This Project**

1. Install dependencies:

```bash
pip install tensorflow numpy matplotlib
```

2. Run the notebook (`.ipynb`) or Python script.

3. The dataset will auto-download via Keras.

---

