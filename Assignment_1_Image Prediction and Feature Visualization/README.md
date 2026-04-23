# **Assignment-1: Image Prediction and Feature Visualization**

**Course Title:** Machine Learning / Deep Learning
**Submitted To:** Sangeeta Biswas
**Submitted By:**
**Name:** Md Nazmul Hasan
**Student ID:** 2111176131
**Department:** Computer Science and Engineering
**University:** University of Rajshahi
**Date:** April 2026

---

## **1. Face Image Collection**

For this task, a dataset of face images was created using a mobile phone camera.

* Images were collected from:

  * Myself
  * Three male participants
  * Three female participants

* Each individual was captured from **five different angles**:

  * Front view
  * Left profile
  * Right profile
  * Slight upward angle
  * Slight downward angle

* **Total images collected:**
  7 persons × 5 angles = **35 images**

* **Ethical consideration:**
  Full consent was obtained from all participants before capturing their images.

---

## **2. Flower Image Collection**

A separate dataset of flower images was collected.

* **Number of flower types:** 5
* **Images per type:** 5
* **Total images:** 25

Examples of flower categories:

* Rose
* Sunflower
* Marigold
* Lily
* Hibiscus

All images were captured using a mobile phone under natural lighting conditions.

---

## **3. Classification Using Pre-trained Models**

Ten pre-trained models from **Keras** were used for classification.

### **Selected Models**

1. VGG16
2. VGG19
3. ResNet50
4. ResNet101
5. MobileNet
6. MobileNetV2
7. InceptionV3
8. Xception
9. DenseNet121
10. EfficientNetB0

---

### **Prediction Tasks**

Each model was used to predict:

* **Top-1 Prediction:** Most probable class
* **Top-5 Predictions:** Top five probable classes

#### **Observations**

* Models trained on ImageNet performed well on **flower classification**
* Face images were often misclassified (since ImageNet is not face-focused)
* Lightweight models (MobileNet) were faster but slightly less accurate
* Deep models (ResNet, DenseNet) gave more stable predictions

---

## **4. Feature Extraction and Visualization**

### **Feature Extraction**

* Features were extracted from the **penultimate layer** of each model
* This resulted in **high-dimensional feature vectors** (e.g., 1024–2048 dimensions)

---

### **Dimensionality Reduction Techniques**

Three techniques were used:

1. Principal Component Analysis (PCA)
2. t-Distributed Stochastic Neighbor Embedding
3. Uniform Manifold Approximation and Projection

---

### **Visualization**

* Features were reduced to **2D space**
* Scatter plots were created for:

  * Face dataset
  * Flower dataset

---

### **Observations**

* **PCA**

  * Fast but less effective in separating complex classes
  * Linear method → limited clustering performance

* **t-SNE**

  * Clear cluster formation
  * Good for visualization but slow

* **UMAP**

  * Best balance between speed and clustering
  * Preserved both global and local structure

---

## **5. Discussion**

### **Model Comparison**

* **Best Feature Separation:**

  * DenseNet121
  * EfficientNetB0

* **Moderate Performance:**

  * ResNet50
  * InceptionV3

* **Lower Performance:**

  * VGG16 / VGG19 (older architectures)

---

### **Why Some Models Perform Better**

* **DenseNet & EfficientNet**

  * Better feature reuse
  * Improved gradient flow
  * Capture fine-grained patterns

* **ResNet**

  * Residual connections help deep learning
  * Good generalization

* **MobileNet**

  * Efficient but less expressive

---

### **Face vs Flower Performance**

* **Flower Images**

  * Better classification accuracy
  * Clear clusters in 2D

* **Face Images**

  * Poor classification (ImageNet limitation)
  * Overlapping clusters

---

## **Conclusion**

This assignment demonstrates that:

* Pre-trained models can effectively classify general objects like flowers
* Feature extraction is powerful for visualization
* UMAP and t-SNE outperform PCA in clustering tasks
* Modern architectures like EfficientNet provide better representations