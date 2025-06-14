# Osteoporosis Classification using Deep Learning
---
## Project Overview
This project focuses on classifying bone conditions using deep learning models trained on X-ray images. The models used include **VGG16, VGG19, InceptionV3, ResNet50, Xception, AlexNet, MobileNetV2 and a Custom CNN**. The goal is to accurately classify images into three categories:

- **Osteopenia** 
- **Osteoporosis** 
- **Normal** 
---
## Dataset
The dataset consists of **X-ray images** of bones, divided into three classes. The images were preprocessed by resizing, normalizing, and augmenting to enhance the model's performance.

|X-ray Images||||Classification|
|----------------------|----------------------|----------------------|----------------------|----------------------|
|![Normal](https://github.com/user-attachments/assets/bdbe54bf-a7f7-45ef-acaa-f54e38c6f6ae)|![Normal](https://github.com/user-attachments/assets/73aef2e4-d48e-4948-b681-39e68e7318c5)|![Normal](https://github.com/user-attachments/assets/92c02e3f-e4bb-4539-a8d4-a22d0190c783)|![Normal](https://github.com/user-attachments/assets/9aa1826a-3e18-482e-9e67-a623b3112462)|Normal|
|![Osteopenia](https://github.com/user-attachments/assets/58838b3b-7d38-478b-8cbd-1df131fc9baf)|![Osteopenia](https://github.com/user-attachments/assets/5bbfb755-1fef-4c97-a125-8f4f1a1428fb)|![Osteopenia](https://github.com/user-attachments/assets/894b6af5-99f4-45ce-9512-7bc04c9e5501)|![Osteopenia](https://github.com/user-attachments/assets/ad5a447d-d298-4cbb-97ff-658bb94bdd55)|Osteopenia|
|![Osteoporosis](https://github.com/user-attachments/assets/7456a0cc-38d0-4f14-a176-f22cdf0a5d63)|![Osteoporosis](https://github.com/user-attachments/assets/adf9ab78-dc4f-4ad7-8a25-cac85940d70a)|![Osteoporosis](https://github.com/user-attachments/assets/e8f2773a-d0a4-4753-a55a-0c434aebbea2)|![Osteoporosis](https://github.com/user-attachments/assets/a450f36f-aced-4f55-82a1-85250a0ccc71)|Osteoporosis|

## Model Architecture
![Model Architecture of Osteoporosis Prediction](https://github.com/user-attachments/assets/f7380181-ab19-41bf-b6c5-f159761a6057)
---
## Models Used
We have trained and evaluated the following deep learning models:
1. **VGG16**
2. **VGG19**
3. **InceptionV3**
4. **ResNet50**
5. **Xception**
6. **AlexNet**
7. **Custom CNN**
8. **Late Fusion**
9. **Dense Net 121**
10. **VGG 16 + VGG 19**
11. **InceptionV3 + XceptionNet**
12. **ResNet 50 + DenseNet 121**

Each model was trained with the same dataset and evaluated using precision, recall, f1-score, accuracy, and confusion matrices.
---
## Performance Metrics
Below is a summary of the classification performance for each model:

| Model       | Accuracy | Precision | Recall | F1-Score | Confusion Matrix | Graphs |
|------------|----------|------------|--------|----------|------------------|--------|
| **VGG16**  | 70%      | 0.74       | 0.70   | 0.70     | ![image](https://github.com/user-attachments/assets/94092c92-4344-4a42-9e8a-ffae9e0ac671) | ![image](https://github.com/user-attachments/assets/7bb28717-0c57-44d5-ac54-a249e8840db5) |
| **VGG19**  | 75%      | 0.77       | 0.75   | 0.75     | ![image](https://github.com/user-attachments/assets/2c4da798-a558-45c6-965f-a7f4f48b1ba9) | ![image](https://github.com/user-attachments/assets/06218327-c539-4199-bd16-ebe6cd3f1969) |
| **InceptionV3** | 88% | 0.89       | 0.88   | 0.88     | ![image](https://github.com/user-attachments/assets/14f0fbe5-bda7-42fc-be21-b55f746739a8) | ![image](https://github.com/user-attachments/assets/af83ce3a-99f3-438a-a8b8-1b4b49f06c93) |
| **ResNet50** | 66%   | 0.74       | 0.66   | 0.65     | ![image](https://github.com/user-attachments/assets/f5fac108-bfa9-4571-b94a-4ff18437a09b) | ![image](https://github.com/user-attachments/assets/cd8b76ad-cea7-4158-a446-4048dcffdaf7) |
| **Xception** | 87%   | 0.88       | 0.87   | 0.87     | ![image](https://github.com/user-attachments/assets/4c5dba18-a3ab-4b58-920d-0bcbfd2053d6) | ![image](https://github.com/user-attachments/assets/9100cc15-e631-4e2a-bc15-5ebb7934e191) |
| **AlexNet** | 85%    | 0.86       | 0.85   | 0.85     | ![image](https://github.com/user-attachments/assets/25120dd6-fa30-412b-b3ef-6b3ae0ed6d6d) | ![image](https://github.com/user-attachments/assets/0ad3578c-8c1a-4f9f-9f37-23640e3211f3) |
|**Late Fusion**|86%|0.86|0.86|0.86|![image](https://github.com/user-attachments/assets/aaadb325-1663-423e-afce-b522d48e88c5)|![image](https://github.com/user-attachments/assets/3ef2e6ce-d88f-47e4-99ef-b9fcf31d528d)|
|**DenseNet 121**|83%|0.83|0.83|0.83|![image](https://github.com/user-attachments/assets/c092727a-2f21-441c-ae98-c0c04349a9aa)|![image](https://github.com/user-attachments/assets/ea0f040c-18db-493a-8bfb-ab028610d744)|
| **MobileNetV2** | 84% | 0.82 | 0.84 | 0.82 | ![image](https://github.com/user-attachments/assets/c7c04cfe-5f43-4d88-9806-7d32f4c4103c) | ![image](https://github.com/user-attachments/assets/6ef85407-5f80-45f7-aba7-c5fe5af90d25)| 
| **Custom CNN** | 89% | 0.89       | 0.89   | 0.89     | ![image](https://github.com/user-attachments/assets/962d3275-17d7-4c9e-86ec-fdf57c72f504) | ![image](https://github.com/user-attachments/assets/1c5899e5-3d42-4aa9-84b3-7d4fedfd570c) |
|*Ensemble Learning*|||||||
|**VGG 16 + VGG 19**|80%|0.81|0.80|0.80|![image](https://github.com/user-attachments/assets/ab18b32f-fef4-43c9-a0c5-719a1d9ee694)||
|**InceptionV3 + Xception**|84%|0.84|0.84|0.84|![image](https://github.com/user-attachments/assets/0a26ff86-a516-45a9-943b-f4c0ddc231cd)||
|**ResNet50 + DenseNet121**|82%|0.82|0.82|0.82|![image](https://github.com/user-attachments/assets/178e1015-a7ed-42f2-809e-e156b15bd54c)||
|**AlexNet + MobileNetV2**|88%|0.89|0.88|0.88|![image](https://github.com/user-attachments/assets/e435e629-8865-42ee-8015-f57f48987816)||
|**InceptionV3 + DenseNet121**|85%|0.86|0.85|0.85|![image](https://github.com/user-attachments/assets/eb79d438-2aef-43a0-aaf5-011b71020002)||
|**Xception + DenseNet121**|85%|0.85|0.85|0.85|![image](https://github.com/user-attachments/assets/7214f53f-68d1-41cc-9571-3f5a1d574cfc)||
|**MobileNetV2 + Xception**|84%|0.85|0.84|0.84|![image](https://github.com/user-attachments/assets/03b94b8f-e297-450d-98f1-ec4374ae9912)||
|**InceptionV3 + MobileNetV2**|84%|0.85|0.84|0.84|![image](https://github.com/user-attachments/assets/270ef149-61f9-415d-8652-c3d0969531ed)||
|**Custom CNN + DenseNet121**|85%|0.85|0.85|0.85|![image](https://github.com/user-attachments/assets/f2dd1ff9-1a2e-49f1-8567-e5333f7c15af)||
---
## Classification Reports
### **VGG16**
```
              precision    recall  f1-score   support

  Osteopenia       0.82      0.55      0.66        75
Osteoporosis       0.60      0.86      0.71       159
      Normal       0.83      0.60      0.70       156

    accuracy                           0.70       390
   macro avg       0.75      0.67      0.69       390
weighted avg       0.74      0.70      0.70       390
```

### **VGG19**
```
              precision    recall  f1-score   support

  Osteopenia       0.81      0.73      0.77        75
Osteoporosis       0.68      0.88      0.77       159
      Normal       0.85      0.63      0.73       156

    accuracy                           0.75       390
   macro avg       0.78      0.75      0.75       390
weighted avg       0.77      0.75      0.75       390
```

### **InceptionV3**
```
              precision    recall  f1-score   support

  Osteopenia       0.85      0.88      0.86        75
Osteoporosis       0.88      0.91      0.89       159
      Normal       0.91      0.87      0.89       156

    accuracy                           0.88       390
   macro avg       0.88      0.88      0.88       390
weighted avg       0.89      0.88      0.88       390
```

### **ResNet50**
```
              precision    recall  f1-score   support

  Osteopenia       0.87      0.35      0.50        75
Osteoporosis       0.57      0.92      0.70       159
      Normal       0.86      0.55      0.67       156

    accuracy                           0.66       390
   macro avg       0.76      0.61      0.62       390
weighted avg       0.74      0.66      0.65       390
```

### **Xception**
```
              precision    recall  f1-score   support

  Osteopenia       0.81      0.89      0.85        75
Osteoporosis       0.89      0.83      0.86       159
      Normal       0.89      0.91      0.90       156

    accuracy                           0.87       390
   macro avg       0.86      0.88      0.87       390
weighted avg       0.88      0.87      0.87       390
```

### **AlexNet**
```
              precision    recall  f1-score   support

  Osteopenia       0.86      0.85      0.86        75
Osteoporosis       0.82      0.88      0.85       159
      Normal       0.89      0.83      0.86       156

    accuracy                           0.85       390
   macro avg       0.86      0.85      0.85       390
weighted avg       0.86      0.85      0.85       390
```

### **Late Fusion**
```
              precision    recall  f1-score   support

  Osteopenia       0.75      0.84      0.79        75
Osteoporosis       0.85      0.85      0.85       159
      Normal       0.93      0.88      0.90       156

    accuracy                           0.86       390
   macro avg       0.84      0.86      0.85       390
weighted avg       0.86      0.86      0.86       390
```

### **DenseNet 121**
```
              precision    recall  f1-score   support

  Osteopenia       0.82      0.87      0.84        75
Osteoporosis       0.79      0.87      0.83       159
      Normal       0.88      0.77      0.82       156

    accuracy                           0.83       390
   macro avg       0.83      0.83      0.83       390
weighted avg       0.83      0.83      0.83       390
```

### **Custom CNN**
```
              precision    recall  f1-score   support

  Osteopenia       0.82      0.75      0.78        75
Osteoporosis       0.87      0.92      0.90       159
      Normal       0.94      0.92      0.93       156

    accuracy                           0.89       390
   macro avg       0.88      0.86      0.87       390
weighted avg       0.89      0.89      0.89       390
```

### **MobileNet V2**
```
              precision    recall  f1-score   support

  Osteopenia       0.73      0.72      0.72        75
Osteoporosis       0.75      0.83      0.79       159
      Normal       0.86      0.78      0.82       156

    accuracy                           0.79       390
   macro avg       0.78      0.78      0.78       390
weighted avg       0.79      0.79      0.79       390
```

## Ensemble Learning

### **VGG 16 + VGG 19**
```
              precision    recall  f1-score   support

  Osteopenia       0.78      0.83      0.80        75
Osteoporosis       0.75      0.86      0.80       159
      Normal       0.89      0.73      0.80       156

    accuracy                           0.80       390
   macro avg       0.81      0.81      0.80       390
weighted avg       0.81      0.80      0.80       390
```

### **InceptionV3 + Xception**
```
              precision    recall  f1-score   support

  Osteopenia       0.78      0.77      0.78        75
Osteoporosis       0.83      0.87      0.85       159
      Normal       0.89      0.85      0.87       156

    accuracy                           0.84       390
   macro avg       0.83      0.83      0.83       390
weighted avg       0.84      0.84      0.84       390
```

### **ResNet50 + DenseNet121**
```
              precision    recall  f1-score   support

  Osteopenia       0.83      0.57      0.68        75
Osteoporosis       0.79      0.92      0.85       159
      Normal       0.85      0.84      0.85       156

    accuracy                           0.82       390
   macro avg       0.82      0.78      0.79       390
weighted avg       0.82      0.82      0.82       390
```
### **AlexNet + MobileNetV2**
```
              precision    recall  f1-score   support

  Osteopenia       0.86      0.81      0.84        75
Osteoporosis       0.85      0.94      0.89       159
      Normal       0.94      0.85      0.89       156

    accuracy                           0.88       390
   macro avg       0.88      0.87      0.87       390
weighted avg       0.89      0.88      0.88       390
```
### **InceptionV3 + DenseNet121**
```
              precision    recall  f1-score   support

  Osteopenia       0.75      0.85      0.80        75
Osteoporosis       0.84      0.86      0.85       159
      Normal       0.93      0.84      0.88       156

    accuracy                           0.85       390
   macro avg       0.84      0.85      0.84       390
weighted avg       0.86      0.85      0.85       390
```
### **Xception + DenseNet121**
```
              precision    recall  f1-score   support

  Osteopenia       0.80      0.88      0.84        75
Osteoporosis       0.81      0.88      0.84       159
      Normal       0.93      0.79      0.86       156

    accuracy                           0.85       390
   macro avg       0.84      0.85      0.84       390
weighted avg       0.85      0.85      0.85       390
```
### **MobileNetV2 + Xception**
```
              precision    recall  f1-score   support

  Osteopenia       0.76      0.85      0.81        75
Osteoporosis       0.79      0.88      0.83       159
      Normal       0.95      0.78      0.86       156

    accuracy                           0.84       390
   macro avg       0.83      0.84      0.83       390
weighted avg       0.85      0.84      0.84       390
```
### **InceptionV3 + MobileNetV2**
```
              precision    recall  f1-score   support

  Osteopenia       0.72      0.80      0.76        75
Osteoporosis       0.82      0.86      0.84       159
      Normal       0.94      0.85      0.89       156

    accuracy                           0.84       390
   macro avg       0.83      0.83      0.83       390
weighted avg       0.85      0.84      0.84       390
```
### **Custom CNN + DenseNet121**
```
              precision    recall  f1-score   support

  Osteopenia       0.83      0.76      0.79        75
Osteoporosis       0.82      0.91      0.86       159
      Normal       0.90      0.83      0.87       156

    accuracy                           0.85       390
   macro avg       0.85      0.84      0.84       390
weighted avg       0.85      0.85      0.85       390
```
---
## Confusion Matrices & Graphs
Each model has an associated **confusion matrix** and **performance graphs** showcasing:
- **Training & Validation Accuracy**
- **Training & Validation Loss**
- **Comparative Model Performance**
---


---
## 🔍 Grad-CAM Heatmap Explanation

Grad-CAM (Gradient-weighted Class Activation Mapping) is a powerful visualization technique used to understand which regions of an input image a Convolutional Neural Network (CNN) focuses on when making predictions.

### 📌 What is Grad-CAM?

Grad-CAM uses the gradients of any target class flowing into the final convolutional layer to generate a **heatmap** that highlights the important regions in the image for prediction. It helps in making deep learning models more interpretable, especially in tasks like image classification.

### 🌈 Heatmap Color Interpretation

The heatmap is overlaid on the original image using a colormap (usually **Jet**), where each color indicates a different level of importance:

| Color            | Importance Level | Description                                               |
| ---------------- | ---------------- | --------------------------------------------------------- |
| 🔵 Blue          | Low              | Regions the model considers less important.               |
| 🟢 Green         | Medium-Low       | Moderately important areas, not critical.                 |
| 🟡 Yellow        | Medium-High      | Areas contributing more to the prediction.                |
| 🔴 Red           | High             | Most influential regions driving the prediction.          |
| 🔸 Purple        | Very Low         | Negligible impact; mostly ignored by the model.           |
| 🟠 Orange / Pink | Medium           | Contributing regions, not the most critical but relevant. |

> 🔥 **Red and Yellow regions** show where the model is "looking" the most while making its decision.

### 📈 Why Use Grad-CAM?

* ✅ Helps validate model behavior
* ✅ Identifies biased or incorrect attention
* ✅ Useful for debugging and model improvement
* ✅ Enhances explainability for sensitive applications (e.g., medical imaging)

### 🖼️ Example Use Cases

* Image classification (e.g., "Is this a cough or not?")
* Object detection
* Medical diagnosis interpretation (e.g., X-ray analysis)

|Grad-CAM Heatmaps|||
|-------------------|-------------------|-------------------|
|![image](https://github.com/user-attachments/assets/af3e3a21-d80a-4a9a-b016-7b15cb326710)|![image](https://github.com/user-attachments/assets/5cc463b6-b649-4f82-a044-55999d18bbb2)|![image](https://github.com/user-attachments/assets/9fd0e0ca-c738-4961-b0aa-25aa6270de79)|
|VGG 16             |VGG 19             |InceptionV3        |
|![image](https://github.com/user-attachments/assets/a16bf5b8-5bac-454b-b93b-c33637030af5)|![image](https://github.com/user-attachments/assets/fdc1d0c1-f194-4a8c-98ed-ae21f9945c5e)|![image](https://github.com/user-attachments/assets/f3c4be31-99ee-4247-9344-163e5128beb5)|
|XceptionNet        |ResNet50           |DenseNet121        |
|![image](https://github.com/user-attachments/assets/83cb1694-c3ea-40e8-9dd0-78c31012089b)|![image](https://github.com/user-attachments/assets/7dbfddef-3e2b-4a7f-8cc1-d9a0dfe0272b)| |
|Late Fusion        |Custom CNN         |                   |

---

## 🚀 Streamlit Web Application 
### Application Overview
We have developed a comprehensive web-based application using Streamlit that provides an intuitive interface for osteoporosis classification. The application loads pre-trained models directly without requiring users to run the entire training pipeline, making it accessible for healthcare professionals and researchers.

### Key Features
### 🎯 Model Selection & Performance

* Multi-Model Support: Choose from 8 different deep learning architectures
* Default Recommendation: InceptionV3 pre-selected (88% accuracy - best performing available model)
* Real-time Switching: Dynamic model loading with performance metrics display
* Model Information: Detailed architecture specs, parameters, and accuracy scores

### 📤 Image Upload & Processing

* Drag-and-Drop Interface: User-friendly file upload with support for PNG, JPG, JPEG formats
* Image Preprocessing: Real-time visualization of image preprocessing steps
* Format Validation: Automatic image format detection and conversion
* Size Optimization: Automatic resizing to model input requirements (224x224 pixels)

🔬 Advanced Classification Results

* Instant Diagnosis: Real-time classification with confidence scores
* Visual Confidence Display: Interactive charts showing probability distribution
* Color-Coded Results:

  * 🟢 Normal: Healthy bone density
  * 🟡 Osteopenia: Moderate bone density loss
  * 🔴 Osteoporosis: Severe bone density loss


* Detailed Metrics: Per-class confidence breakdown with visual progress bars

### 📊 Interactive Visualizations

* Confidence Charts: Plotly-based interactive bar charts for classification probabilities
* Prediction History: Track and visualize prediction patterns over time
* Model Comparison: Side-by-side performance analysis capabilities
* Real-time Updates: Dynamic chart updates with each new prediction

### 🎨 Modern UI/UX Design

* Professional Interface: Medical-grade design with gradient themes
* Responsive Layout: Optimized for desktop and tablet devices
* Intuitive Navigation: Clean sidebar with organized controls
* Educational Content: Built-in information cards explaining bone conditions

## Installation & Setup
### Prerequisites
* pip install streamlit tensorflow plotly pandas opencv-python pillow numpy *

### Model Files Required
Place the following trained model files in the application directory:

* AlexNet_knee_osteo_model.keras
* DenseNet121_osteo_model.keras
* InceptionV3_knee_osteo_model.keras (Recommended)
* MobileNetV2_knee_osteo_model.keras
* ResNet50_knee_osteo_model.keras
* VGG16_knee_osteo_model.keras
* VGG19_knee_osteo_model.keras
* Xception_knee_osteo_model.keras

### Running the Application
streamlit run app.py

## Application Workflow
![image alt](https://github.com/Durga200422/Osteoporosis-Detection/blob/858f76379ffbef4f076bd5d19d57a5d13fc6bd6d/Application%20Interface.png)

1. Model Selection: Choose from 8 available deep learning models (InceptionV3 recommended)
2. Image Upload: Upload knee X-ray image via drag-and-drop or file browser
3. Preprocessing: Automatic image preprocessing with optional visualization
4. Classification: Real-time bone condition analysis with confidence scores
5. Results Display:

   * Primary diagnosis with confidence percentage
   * Detailed probability breakdown for all classes
   * Interactive confidence visualization charts

6. History Tracking: Monitor prediction patterns and model performance over time

## Technical Specifications

* Framework: Streamlit 1.28+
* Deep Learning: TensorFlow/Keras 2.13+
* Visualization: Plotly 5.15+
* Image Processing: OpenCV, PIL
* Input Format: 224x224 RGB images
* Output Classes: Normal, Osteopenia, Osteoporosis
* Performance: Cached model loading for optimal speed

## Use Cases

* Medical Research: Compare different model architectures for * osteoporosis detection
* Clinical Decision Support: Assist healthcare professionals in * bone density assessment
* Educational Tool: Demonstrate deep learning applications in * medical imaging
* Model Validation: Test and validate trained models on new * datasets
* Screening Tool: Preliminary assessment for bone health evaluation

## Safety & Disclaimer
⚠️ Important: This application is designed for research and educational purposes only. All predictions should be validated by qualified healthcare professionals. The tool is not intended to replace professional medical diagnosis or treatment decisions.


## Conclusion

Among all models, **Custom CNN** performed the best with **89% accuracy**, followed by **InceptionV3** at **88%**. The **VGG and ResNet architectures** showed moderate performance. The **confusion matrices and graphs** provide further insights into model performance.
---
## Authors
- **[Narapureddy Durga Prasad Reddy](https://www.linkedin.com/in/narapureddy-d-2a5402252/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)** - AI/ML Researcher
---
*This project was developed as part of an ongoing research initiative in medical image classification using deep learning.*
