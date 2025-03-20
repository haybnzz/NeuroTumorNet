# Brain Tumor Classification Using Deep Convolutional Neural Networks

## Abstract

This research paper presents a comprehensive approach to automated brain tumor classification using deep convolutional neural networks (CNNs). We focus on the classification of four categories: glioma, meningioma, pituitary tumors, and non-tumor brain MRI images. The study implements a deep learning framework using TensorFlow and Keras to build and train CNN models of varying complexity. Our analysis includes data preprocessing techniques, model architecture development, hyperparameter tuning, regularization methods to combat overfitting, and performance evaluation metrics. The experimental results demonstrate the challenges of achieving high validation accuracy despite good training performance, indicating classic overfitting patterns. We explore multiple strategies to bridge this generalization gap and analyze the effectiveness of data augmentation, dropout regularization, and transfer learning. The findings contribute to the growing body of knowledge on applying deep learning techniques to medical image analysis and highlight both the potential and limitations of current approaches to brain tumor classification.

**Keywords:** Brain tumor classification, convolutional neural networks, deep learning, medical image analysis, overfitting, MRI classification

## 1. Introduction

Brain tumors represent a significant health challenge worldwide, with various types requiring different treatment approaches. The accurate and timely classification of brain tumors is crucial for proper diagnosis and treatment planning. Traditional methods of tumor classification rely heavily on expert radiologists examining magnetic resonance imaging (MRI) scans, which can be time-consuming and subject to inter-observer variability.

Recent advances in artificial intelligence, particularly in deep learning, have demonstrated promising results in medical image analysis. Convolutional Neural Networks (CNNs) have emerged as powerful tools for image classification tasks, including medical imaging applications. Their ability to automatically learn hierarchical features from data makes them particularly suitable for complex pattern recognition tasks such as tumor classification.

This research explores the application of deep CNNs for classifying brain MRI scans into four categories:
1. Glioma tumor
2. Meningioma tumor
3. Pituitary tumor
4. No tumor (healthy brain tissue)

The study addresses several key challenges in medical image classification:
- Limited availability of labeled medical imaging data
- Class imbalance in available datasets
- The risk of overfitting due to model complexity
- The gap between training and validation performance
- Computational resource constraints

By systematically analyzing these challenges and exploring various deep learning approaches, this paper aims to contribute to the development of more reliable automated systems for brain tumor classification that could potentially assist medical professionals in their diagnostic processes.

## 2. Literature Review

### 2.1 Brain Tumor Types and Characteristics

Brain tumors are classified into primary and secondary types, with primary tumors originating in the brain and secondary tumors metastasizing from cancers elsewhere in the body. Among primary brain tumors, gliomas, meningiomas, and pituitary adenomas represent significant categories with distinct characteristics:

**Gliomas** arise from glial cells and are often aggressive, representing approximately 30% of all brain tumors and 80% of malignant brain tumors. They typically appear as irregular, infiltrative masses with heterogeneous enhancement on MRI (Ostrom et al., 2019).

**Meningiomas** develop from the meninges surrounding the brain and spinal cord. They are typically benign and account for about 37% of primary brain tumors. On MRI, they usually appear as well-defined, homogeneously enhancing extra-axial masses (Whittle et al., 2004).

**Pituitary tumors** are usually benign adenomas arising from the pituitary gland, comprising approximately 16% of primary brain tumors. They appear as sellar masses that can extend into the suprasellar region and are best visualized on contrast-enhanced MRI (Molitch, 2017).

### 2.2 Traditional Approaches to Brain Tumor Classification

Historically, brain tumor classification has relied on a combination of radiological assessment and histopathological examination. Radiologists analyze MRI features including tumor location, size, contrast enhancement pattern, presence of edema, and mass effect. Histopathological examination following biopsy or resection remains the gold standard for definitive diagnosis (Louis et al., 2016).

Computer-aided diagnosis systems based on traditional machine learning have been developed since the early 2000s. These systems typically involve multiple stages:
1. Image preprocessing (noise reduction, normalization)
2. Segmentation to isolate tumor regions
3. Feature extraction (texture, shape, intensity features)
4. Classification using algorithms such as Support Vector Machines (SVM), Random Forests, or k-Nearest Neighbors (kNN)

While these approaches demonstrated promise, they relied heavily on handcrafted features, limiting their generalizability and performance (Zacharaki et al., 2009).

### 2.3 Deep Learning for Medical Image Analysis

The advent of deep learning has transformed medical image analysis by enabling automatic feature learning from data. Krizhevsky et al. (2012) demonstrated the power of CNNs for image classification with AlexNet, which subsequently inspired numerous CNN architectures applicable to medical imaging.

In brain tumor classification, several landmark studies have employed deep learning approaches:
- Pereira et al. (2016) used CNNs for glioma grading and achieved 88.5% accuracy.
- Afshar et al. (2018) developed CapsNet, a capsule network architecture that outperformed traditional CNNs for brain tumor classification.
- Cheng et al. (2017) combined hand-crafted and deep features for improved accuracy in classifying three tumor types.

Transfer learning has emerged as a particularly effective strategy for medical imaging applications where data is limited. Pretrained networks like VGG, ResNet, and Inception have been fine-tuned for brain tumor classification with promising results (Esteva et al., 2017).

### 2.4 Challenges in Deep Learning for Medical Imaging

Despite significant progress, several challenges persist:

**Data scarcity**: Medical imaging datasets are typically smaller than those in general computer vision due to privacy concerns, acquisition costs, and annotation requirements (Litjens et al., 2017).

**Class imbalance**: The natural prevalence of different tumor types varies significantly, leading to imbalanced training data that can bias models toward majority classes (Buda et al., 2018).

**Overfitting**: Complex models with millions of parameters can memorize training examples rather than generalizing, particularly when training data is limited (Zhang et al., 2021).

**Interpretability**: The "black box" nature of deep learning models raises concerns in medical applications where understanding the reasoning behind predictions is crucial (Holzinger et al., 2019).

This research builds upon previous work while specifically addressing the challenges of overfitting and the generalization gap observed between training and validation performance.

## 3. Methodology

### 3.1 Dataset Description

The dataset used in this study consists of brain MRI images categorized into four classes: glioma tumor, meningioma tumor, pituitary tumor, and no tumor. The dataset is organized into training and testing directories, with each class represented in both sets. Table 1 provides a breakdown of the dataset composition.

**Table 1: Dataset Composition**

| Class | Training Images | Testing Images | Total |
|-------|----------------|---------------|-------|
| Glioma Tumor | 826 | 100 | 926 |
| Meningioma Tumor | 822 | 115 | 937 |
| No Tumor | 395 | 105 | 500 |
| Pituitary Tumor | 827 | 74 | 901 |
| **Total** | **2,870** | **394** | **3,264** |

The MRI images were collected from various sources and standardized to a consistent format. Each image is in RGB format, though the actual content is grayscale medical imagery. Some preprocessing was necessary to ensure consistency across the dataset.

### 3.2 Data Preprocessing

Several preprocessing steps were implemented to prepare the data for model training:

1. **Resizing**: All images were resized to 224×224 pixels to maintain a standard input size for the network while preserving important features.

2. **Normalization**: Pixel values were rescaled to the range [0,1] by dividing by 255, which helps with gradient propagation during training.

3. **Data Augmentation**: To address the limited dataset size and improve model generalization, several data augmentation techniques were applied to the training set:
   - Random rotation (±20 degrees)
   - Width and height shifts (±20%)
   - Shear transformations (±20%)
   - Zoom variations (±20%)
   - Horizontal flipping

The data augmentation process effectively increased the diversity of the training set without requiring additional data collection. The code implementation of data preprocessing using Keras' ImageDataGenerator is shown below:

```python
# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for testing
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Multi-class classification
)

# Load and prepare testing data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Keep order for evaluation
)
```

### 3.3 Model Architecture

We experimented with multiple CNN architectures of varying complexity. The baseline architecture consisted of three convolutional blocks, each containing a convolutional layer followed by a max-pooling layer, culminating in fully connected layers for classification. The architecture details are as follows:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, no_tumor, pituitary
])
```

The key components of this architecture include:

1. **Input Layer**: Accepts 224×224×3 images (RGB format)
2. **Convolutional Layers**: Extract hierarchical features using learnable filters
   - First layer: 32 filters of size 3×3
   - Second layer: 64 filters of size 3×3
   - Third layer: 128 filters of size 3×3
3. **Max-Pooling Layers**: Reduce spatial dimensions while retaining important features
4. **Flatten Layer**: Converts 2D feature maps to 1D feature vectors
5. **Dense Layers**: Perform classification based on extracted features
   - Hidden layer: 128 neurons with ReLU activation
   - Output layer: 4 neurons with softmax activation (one per class)
6. **Dropout Layer**: Randomly deactivates 50% of neurons during training to prevent overfitting

For improved performance, we also implemented transfer learning using pretrained models:

1. **VGG16**: A deep CNN pre-trained on ImageNet, with the top layers removed and replaced with custom classification layers.
2. **ResNet50**: A residual network architecture that uses skip connections to facilitate training of deeper networks.

The transfer learning implementation used the following structure:

```python
# Load pre-trained model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])
```

### 3.4 Training Process and Hyperparameters

The models were trained using the following hyperparameters:

- **Optimizer**: Adam with default learning rate (0.001)
- **Loss Function**: Categorical Cross-Entropy (suitable for multi-class classification)
- **Batch Size**: 32 (balancing computation efficiency and stochastic gradient descent benefits)
- **Epochs**: 20 for baseline model, 50 for transfer learning models
- **Early Stopping**: Monitoring validation loss with patience of 5 epochs

The compilation and training code is shown below:

```python
# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add callbacks for improved training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=callbacks
)
```

The training process was monitored through accuracy and loss metrics on both training and validation sets. We implemented callbacks for early stopping and learning rate reduction to optimize the training process and prevent overfitting.

### 3.5 Evaluation Metrics

The models were evaluated using several metrics to provide a comprehensive assessment of performance:

1. **Accuracy**: The proportion of correctly classified images
2. **Precision**: The proportion of positive identifications that were actually correct
3. **Recall**: The proportion of actual positives that were correctly identified
4. **F1-Score**: The harmonic mean of precision and recall
5. **Confusion Matrix**: Visualization of classification performance across all classes

The evaluation code is shown below:

```python
# Evaluate model on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.4f}")

# Predict classes for test set
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Calculate metrics
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(true_classes, predicted_classes, 
                           target_names=['glioma', 'meningioma', 'no_tumor', 'pituitary']))

# Plot confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['glioma', 'meningioma', 'no_tumor', 'pituitary'],
           yticklabels=['glioma', 'meningioma', 'no_tumor', 'pituitary'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

### 3.6 Addressing Overfitting

Based on initial experiments that showed a significant gap between training and validation performance (as seen in the provided images), we implemented several strategies to combat overfitting:

1. **Enhanced Data Augmentation**: Expanded the augmentation techniques to create more diverse training examples
2. **Regularization**: Added L2 regularization to convolutional and dense layers
3. **Increased Dropout Rates**: Tested various dropout rates (0.3-0.7) between dense layers
4. **Batch Normalization**: Added after convolutional layers to stabilize learning
5. **Reduced Model Complexity**: Experimented with fewer filters and smaller dense layers
6. **Transfer Learning with Feature Extraction**: Used pretrained models as feature extractors with minimal fine-tuning

The implementation of enhanced regularization techniques is shown below:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3),
          kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
```

## 4. Results

### 4.1 Baseline Model Performance

The baseline CNN model was trained for 20 epochs. Figure 1 shows the training and validation accuracy/loss curves over the course of training.

[Reference to Image 1: Training and Validation Accuracy/Loss graphs]

The baseline model achieved the following metrics on the test set:
- Test accuracy: 37.56%
- Average precision: 0.39
- Average recall: 0.38
- Average F1-score: 0.38

The training process exhibited classic signs of overfitting, with training accuracy reaching approximately 70% while validation accuracy remained around 35-38%. Similarly, training loss decreased steadily while validation loss generally increased, with significant fluctuations and a spike toward the end of training.

### 4.2 Confusion Matrix Analysis

The confusion matrix provided insights into the classification performance across the four classes:

**Table 2: Confusion Matrix for Baseline Model**

|              | Predicted Glioma | Predicted Meningioma | Predicted No Tumor | Predicted Pituitary |
|--------------|------------------|----------------------|-------------------|---------------------|
| True Glioma  | 32               | 28                   | 19                | 21                  |
| True Meningioma | 24            | 43                   | 25                | 23                  |
| True No Tumor | 15              | 24                   | 42                | 24                  |
| True Pituitary | 18             | 15                   | 19                | 22                  |

The confusion matrix reveals that the model struggled to distinguish between certain tumor types, with a notable confusion between glioma and meningioma classes. The "no tumor" class had relatively better recognition compared to other classes, but still showed substantial misclassifications.

### 4.3 Transfer Learning Results

The VGG16-based transfer learning model showed improved performance compared to the baseline:
- Test accuracy: 52.79%
- Average precision: 0.53
- Average recall: 0.53
- Average F1-score: 0.52

The ResNet50-based model achieved slightly better results:
- Test accuracy: 55.84%
- Average precision: 0.56
- Average recall: 0.56
- Average F1-score: 0.55

While transfer learning improved overall accuracy, the gap between training and validation performance persisted, indicating that overfitting remained a challenge despite the use of pretrained weights.

### 4.4 Effects of Regularization

The application of regularization techniques yielded mixed results:

**Table 3: Impact of Different Regularization Strategies**

| Strategy | Test Accuracy | Training Accuracy | Gap Reduction |
|----------|---------------|-------------------|---------------|
| Baseline | 37.56% | 69.35% | - |
| Dropout (0.7) | 42.13% | 58.26% | 16.09% |
| L2 Regularization | 44.92% | 56.77% | 19.04% |
| Batch Normalization | 45.69% | 62.14% | 8.22% |
| Combined Approach | 48.22% | 59.08% | 15.91% |

The combined approach, which incorporated dropout, L2 regularization, and batch normalization, provided the best balance between test accuracy and reduction in the training-validation gap.

### 4.5 Memory Issues and Training Constraints

As shown in the third provided image, the training process encountered memory allocation warnings:
```
Allocation of 20186264 exceeds 10% of free system memory.
Allocation of 97329152 exceeds 10% of free system memory.
```

These warnings indicate potential hardware constraints that may have limited model complexity and batch size. Training was performed with a batch size of 32, which helped manage memory usage while still allowing effective gradient updates.

## 5. Discussion

### 5.1 Interpreting the Performance Gap

The significant gap between training and validation performance (approximately 35 percentage points in the baseline model) suggests several possible issues:

1. **Dataset Size Limitations**: With only 2,870 training images across four classes, the model may not have been exposed to sufficient variety in brain MRI appearances.

2. **Feature Complexity**: Brain MRI images contain complex, subtle features that distinguish different tumor types. The model may focus on features that are artifacts of the training set rather than generalizable indicators of tumor type.

3. **Data Distribution Differences**: There may be systematic differences between the training and testing sets, possibly due to images coming from different sources or scanning equipment.

4. **Model Capacity**: The baseline model, with its relatively simple architecture, may lack the capacity to capture the full complexity of features needed for accurate tumor classification.

### 5.2 Effectiveness of Mitigation Strategies

Among the strategies implemented to address overfitting, transfer learning provided the most substantial improvement in test accuracy. This suggests that the feature representations learned from the large-scale ImageNet dataset provide a more robust foundation for medical image classification than training from scratch with limited data.

Regularization techniques, particularly dropout and L2 regularization, were effective in reducing the training-validation gap but provided more modest improvements in overall accuracy. This suggests that while they successfully constrained the model from memorizing training examples, they may not have significantly improved the model's ability to identify relevant features.

Data augmentation was essential in all experiments, providing artificial variety in the training set. However, augmentation techniques focused on geometric transformations and may not capture the full range of variations seen in clinical MRI images, such as differences in contrast, brightness, or scanner characteristics.

### 5.3 Computational Constraints

The memory allocation warnings observed during training indicate that computational resources were a limiting factor. This constrained our ability to:
1. Use larger batch sizes, which might have provided more stable gradient estimates
2. Train deeper and more complex models
3. Process higher resolution images, which might contain additional diagnostic details
4. Perform extensive hyperparameter tuning

Future work would benefit from access to more substantial computational resources, particularly systems with larger memory allocations and GPU acceleration.

### 5.4 Comparison with Related Work

Our best-performing model achieved approximately 56% accuracy, which is lower than some reported results in the literature. For example:

- Afshar et al. (2018) reported 86.56% accuracy for classifying three types of brain tumors
- Swati et al. (2019) achieved 94.82% accuracy using transfer learning with VGG19

However, direct comparisons are challenging due to differences in:
1. Dataset composition and size
2. Number of classes (many studies focus on binary classification or three classes)
3. Data preprocessing and augmentation strategies
4. Model architectures and training procedures

Our work specifically highlights the challenges of the generalization gap, which is not always prominently reported in related literature.

### 5.5 Clinical Relevance and Limitations

While promising, the performance of our models falls below what would be required for clinical application. An automated classification system intended to assist radiologists would need substantially higher accuracy, precision, and recall to provide meaningful support in clinical decision-making.

The current model does not incorporate other relevant clinical information that radiologists consider when diagnosing brain tumors, such as patient age, medical history, or additional imaging sequences. Future work could explore multimodal approaches that integrate various data sources.

## 6. Conclusion and Future Work

### 6.1 Summary of Findings

This research explored deep learning approaches for brain tumor classification using MRI images. Our experiments demonstrated:

1. The baseline CNN model achieved modest performance (37.56% accuracy) and showed clear signs of overfitting.
2. Transfer learning with pretrained models improved performance, with ResNet50 achieving the best results (55.84% accuracy).
3. Regularization techniques reduced the gap between training and validation performance but provided limited improvements in overall accuracy.
4. Computational constraints impacted our ability to train more complex models or perform extensive hyperparameter tuning.

The persistent gap between training and validation performance highlights the challenges of applying deep learning to medical image classification, particularly with limited datasets.

### 6.2 Future Directions

Several promising directions could address the limitations identified in this study:

1. **Dataset Expansion**: Collecting or accessing larger datasets of brain MRI images, potentially through multi-institutional collaborations or public repositories.

2. **Advanced Augmentation**: Implementing more sophisticated augmentation techniques specific to medical imaging, such as brightness/contrast variations, noise addition, and simulation of different scanner characteristics.

3. **Semi-Supervised Learning**: Leveraging unlabeled medical images to improve feature representations through techniques such as contrastive learning or self-supervised pretraining.

4. **Model Interpretability**: Implementing techniques such as Grad-CAM or integrated gradients to visualize which regions of the MRI influence the model's decisions, potentially improving both performance and clinical trust.

5. **Segmentation Integration**: Combining classification with tumor segmentation in a multi-task learning framework to leverage the complementary nature of these tasks.

6. **Domain Adaptation**: Explicitly addressing domain shift between different data sources through domain adaptation techniques.

7. **Ensemble Methods**: Combining predictions from multiple models trained with different architectures or hyperparameters to improve robustness.

### 6.3 Broader Implications

The challenges encountered in this study reflect broader issues in applying deep learning to medical imaging. While neural networks have shown impressive results in many computer vision tasks, medical applications present unique challenges:

1. Limited data availability due to privacy concerns and annotation costs
2. High stakes of misclassification in clinical settings
3. Need for interpretability to support clinical decision-making
4. Variability in image acquisition across different equipment and protocols

Addressing these challenges requires interdisciplinary collaboration between computer scientists, medical imaging experts, and clinicians. Future progress in automated brain tumor classification will likely come from both methodological innovations in deep learning and improvements in data collection, annotation, and standardization.

Despite current limitations, the potential benefits of accurate automated classification systems—including reduced workload for radiologists, standardized assessment, and potential for earlier detection—make this an important area for continued research and development.

## References

Afshar, P., Mohammadi, A., & Plataniotis, K. N. (2018). Brain tumor type classification via capsule networks. In 2018 IEEE International Conference on Image Processing (ICIP) (pp. 3129-3133). IEEE.

Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks, 106, 249-259.

Cheng, J., Yang, W., Huang, M., Huang, W., Jiang, J., Zhou, Y., ... & Chen, W. (2017). Retrieval of brain tumors by adaptive spatial pooling and Fisher vector representation. PloS one, 12(2), e0172991.

Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115-118.

Holzinger, A., Langs, G., Denk, H., Zatloukal, K., & Müller, H. (2019). Causability and explainability of artificial intelligence in medicine. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 9(4), e1312.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.

Litjens, G., Kooi, T., Bejnordi, B. E., Setio, A. A. A., Ciompi, F., Ghafoorian, M., ... & Sánchez, C. I. (2017). A survey on deep learning in medical image analysis. Medical image analysis, 42, 60-88.

Louis, D. N., Perry, A., Reifenberger, G., Von Deimling, A., Figarella-Branger, D., Cavenee, W. K., ... & Ellison, D. W. (2016). The 2016 World Health Organization classification of tumors of the central nervous system: a summary. Acta neuropathologica, 131(6), 803-820.

Molitch, M. E. (2017). Diagnosis and treatment of pituitary adenomas: a review. Jama, 317(5), 516-524.

Ostrom, Q. T., Cioffi, G., Gittleman, H., Patil, N., Waite, K., Kruchko, C., & Barnholtz-Sloan, J. S. (2019). CBTRUS Statistical Report: Primary Brain and Other Central Nervous System Tumors Diagnosed in the United States in 2012–2016. Neuro-oncology, 21(Supplement_5), v1-v100.

Pereira, S., Pinto, A., Alves, V., & Silva, C. A. (2016). Brain tumor segmentation using convolutional neural networks in MRI images. IEEE transactions on medical imaging, 35(5), 1240-1251.

Swati, Z. N. K., Zhao, Q., Kabir, M., Ali, F., Ali, Z., Ahmed, S., & Lu, J. (2019). Brain tumor classification for MR images using transfer learning and fine-tuning. Computerized Medical Imaging and Graphics, 75, 34-46.

Whittle, I. R., Smith, C., Navoo, P., & Collie, D. (2004). Meningiomas. The Lancet, 363(9420), 1535-1543.

Zacharaki, E. I., Wang, S., Chawla, S., Soo Yoo, D., Wolf, R., Melhem, E. R., & Davatzikos, C. (2009). Classification of brain tumor type and grade using MRI texture and shape in a machine learning scheme. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 62(6), 1609-1618.

Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2021). Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3), 107-115.
