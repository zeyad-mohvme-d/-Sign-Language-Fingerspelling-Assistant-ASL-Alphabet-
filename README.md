# ✋ Sign Language Fingerspelling Assistant (ASL Alphabet)

This project is an AI-based system for recognizing **American Sign Language (ASL) fingerspelling letters** using deep learning.  
Multiple neural network architectures were trained, evaluated, and deployed in an interactive web application.

---

## 📌 Dataset

### Dataset Link  
https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data

### Dataset Description  
The ASL Alphabet dataset is a collection of images representing hand signs for the American Sign Language alphabet.

- Total classes: **29**
  - 26 alphabet letters (A–Z)
  - 3 special classes: **SPACE**, **DELETE**, **NOTHING**
- Training set:
  - ~87,000 images
  - Image size: 200×200 pixels
- Test set:
  - 29 images (one image per class)

The special classes (SPACE, DELETE, NOTHING) are useful for real-time applications and sentence-level recognition systems.

---

## 🧠 Models

Four different deep learning architectures were trained and compared:

### 1️⃣ Custom CNN  
- Epochs: 30  
- Training Accuracy: **0.9670**  
- Validation Accuracy: **0.9734**

![CNN Accuracy](https://github.com/user-attachments/assets/2c01df2a-d463-4047-b475-67f7e3d9f2e8)

---

### 2️⃣ EfficientNetB0  
- Early stopping at 19 epochs  
- Training Accuracy: **0.9325**  
- Validation Accuracy: **0.9672**

![EfficientNetB0 Accuracy](https://github.com/user-attachments/assets/d8d4a6f5-387d-4d61-a778-1f0e21ba0039)

---

### 3️⃣ EfficientNetB0_V2  
- Early stopping at 19 epochs  
- Training Accuracy: **0.9925**  
- Validation Accuracy: **0.9872**

![EfficientNetB0_V2 Accuracy](https://github.com/user-attachments/assets/7b2d4604-b605-4f0a-a865-1fe25b94d4e2)

---

### 4️⃣ ResNet50  
- Epochs: 10  
- Training Accuracy: **0.9827**  
- Validation Accuracy: **0.9920**

![ResNet50 Accuracy](https://github.com/user-attachments/assets/a5c9f177-96ed-4e22-9371-47f14b0fa1e8)

---

## 📊 Evaluation Notes

Although the models achieved high training and validation accuracy, evaluation on the official test set results in low and similar metric values across models.

This is expected because:
- The test set contains only **29 images**
- Each class has only one test image
- Such a small test set is not statistically sufficient for reliable quantitative evaluation

Therefore, model performance is primarily assessed using:
- Training accuracy
- Validation accuracy
- Qualitative evaluation on real-world images and live webcam input

---

## 🖥️ Deployment

The models were deployed using **Streamlit** to create a simple and interactive graphical user interface (GUI).

The GUI allows users to:
- Upload an image of an ASL hand sign
- Use a live webcam for real-time prediction
- Select between multiple trained models
- View prediction confidence scores
- Compare predictions across models

This makes the system suitable for demonstration and real-time usage.

---

## 🔍 Explainability (Grad-CAM)

To improve interpretability, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was integrated for transfer learning models such as:
- EfficientNetB0_V2
- ResNet50

Grad-CAM generates a heatmap highlighting the regions of the input image that contributed most to the model’s prediction, ensuring that predictions are based on the hand gesture rather than background noise.

---

## 📁 Project Structure

