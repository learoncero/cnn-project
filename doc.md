# Experiment Results: CNN Enhancements and Strategies for Emotion Recognition

## Objective
To explore and evaluate various CNN architectures, pooling strategies, and feature enhancement techniques to optimize the performance of emotion recognition using grayscale images.

---

## Experimental Setup
- **Dataset**: Emotion recognition dataset with 48x48 grayscale images.
- **Model Architecture**: CNN with various enhancements and strategies tested.
- **Pooling Types**:
    - Max Pooling (`nn.MaxPool2d`)
    - Avg Pooling (`nn.AvgPool2d`)
- **Other Techniques...**
- **Augmentation**: Data augmentation techniques such as rotation, scaling, and flipping.
- **Evaluation Metric**: Model accuracy on validation/test set.
- **Loss Function**: `nn.CrossEntropyLoss()` for multi-class classification.
- **Optimizer**: `optim.Adam` with a learning rate of `0.001` or `0.0001`.
- **Hardware**: GPU-enabled training environment (if available).

---

## Results

### 1. Varying the Number of Convolutional Layers

| **Number of Layers** | **Validation Accuracy** | **Observations**                                                                                        | **Notebook**                                                                                                                                                           |
|----------------------|-------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **4 Layers**         | 0.79                    | Model showed underfitting, likely due to insufficient feature extraction.                               | 2_Group17_DLProject.ipynb                                                                                                                                              |
| **5 Layers**         | 0.83                    | Better accuracy but limited capacity to extract deep features.                                          | 1_Group17_DLProject.ipynb (Base-line Notebook)                                                                                                                         |
| **6 Layers**         | 0.84                    | Highest accuracy, indicating a balance between feature extraction and model complexity.                 | 3_Group17_DLProject.ipynb / 5_Group17_DLProject.ipynb / 7_Group17_DLProject.ipynb / 8_Group17_DLProject.ipynb / 9_Group17_DLProject.ipynb / 10_Group17_DLProject.ipynb |
| **7 Layers**         | 0.83                    | Marginal decrease in accuracy, possibly due to overfitting or redundancy. Or a threshhold has been hit. | 4_Group17_DLProject.ipynb                                                                                                                                              |

### 2. Pooling Strategies

| **Pooling Type**           | **Validation Accuracy** | **Observations**                                                                                                                | **Notebook**                                                                                                                                                           |
|----------------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Max Pooling**            | 0.84                    | Preserves key features while reducing spatial dimensions effectively.                                                           | 2_Group17_DLProject.ipynb / 3_Group17_DLProject.ipynb / 4_Group17_DLProject.ipynb                                                                                      |
| **Avg Pooling**            | 0.85                    | Smoother feature extraction.                                                                                                    | 5_Group17_DLProject.ipynb / 6_Group17_DLProject.ipynb / 7_Group17_DLProject.ipynb / 8_Group17_DLProject.ipynb / 9_Group17_DLProject.ipynb / 10_Group17_DLProject.ipynb |
| **Global Average Pooling** | 0.83                    | Reduces the spatial dimensions to a single value per feature map, which can help in reducing overfitting and model complexity.  | 6_Group17_DLProject.ipynb                                                                                                                                              |


### 3. Other Techniques

| **Technique**       | **Validation Accuracy** | **Notes**                                             | **Notebook**               |
|---------------------|-------------------------|-------------------------------------------------------|----------------------------|
| **Dilated Layers**  | 0.83                    | Added a dilation values to conv layers                | 8_Group17_DLProject.ipynb  |
| **GeLU Function**   | 0.84                    | Implemented GeLU Activation Functions instead of ReLU | 9_Group17_DLProject.ipynb  |
| **Residual Blocks** |                         |                                                       | 10_Group17_DLProject.ipynb |

### 4. Data Augmentation

| **Technique**    | **Validation Accuracy** | **Notes**                                                                         | **Notebook**              |
|------------------|-------------------------|-----------------------------------------------------------------------------------|---------------------------|
| **Augmentation** | 0.93                    | Augmented data (such as horizontal flip, crop, rotation) was added to the dataset | 7_Group17_DLProject.ipynb |


---

## Conclusion
- **Optimal Model Configuration**:
    - **6 convolutional layers**
    - **Avg Pooling** as the pooling strategy.
    - **Global Average Pooling** to reduce model complexity.
    - **Data Augmentation** for improved generalization.
- **Key Insight**: Increasing the number of convolutional layers improves accuracy up to a point, but too many layers can lead to diminishing returns or overfitting. Avg Pooling can be more effective in capturing general features compared to Max Pooling. Data augmentation is crucial for training deep learning models with limited data.
