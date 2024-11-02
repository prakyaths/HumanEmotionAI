# Human Emotion Detection Using LeNet

## Overview
This project implements a Convolutional Neural Network (CNN) based on the LeNet architecture to classify human emotions from images. The model is trained to recognize three basic emotions: *Angry*, *Happy*, and *Sad*.

## Project Structure

| Directory/File                      | Description                                     |
|-------------------------------------|-------------------------------------------------|
| `data/`                             | Directory to store the emotion dataset          |
| `models/`                           | Directory to save trained models                |
| `src/`                              | Source code for model building and training     |
| ├── `Human_Emotion_Detection.ipynb` | Main Colab notebook for the project             |
| `results/`                          | Directory to save evaluation metrics and plots  |
| `README.md`                         | Project documentation (this file)               |

## Dataset
This project uses an emotion recognition dataset containing labeled images representing three classes of human emotions:

- **Angry**
- **Happy**
- **Sad**

Each image is resized to 256x256 pixels for input into the model.

## Model Architecture
The model is based on the **LeNet** CNN architecture, customized with additional layers and batch normalization. The configuration includes parameters such as dropout rate and regularization rate to help control overfitting.

### Model Architecture in Detail:

| Layer               | Description                                             |
|---------------------|---------------------------------------------------------|
| Input               | 256x256 RGB images                                      |
| Conv2D              | 6 filters, kernel size 3x3, stride 1, ReLU activation   |
| BatchNormalization  | Normalizes the activations                              |
| MaxPooling2D        | Pool size 2x2, stride 2                                 |
| Dropout             | Dropout rate of 0.0                                     |
| Conv2D              | 16 filters, kernel size 3x3, stride 1, ReLU activation  |
| BatchNormalization  | Normalizes the activations                              |
| MaxPooling2D        | Pool size 2x2, stride 2                                 |
| Flatten             | Reshapes the 2D matrices into 1D vectors                |
| Dense               | 100 units, ReLU activation                              |
| BatchNormalization  | Normalizes the activations                              |
| Dropout             | Dropout rate of 0.0                                     |
| Dense               | 10 units, ReLU activation                               |
| BatchNormalization  | Normalizes the activations                              |
| Output (Softmax)    | 3 units (Angry, Happy, Sad)                             |

## Training Configuration
The model is trained using the **Adam optimizer** with a learning rate of `1e-3`. The training parameters are as follows:

- **Batch Size**: 32
- **Image Size**: 256x256
- **Epochs**: 20
- **Dropout Rate**: 0.0
- **Regularization Rate**: 0.0
- **Data Augmentation**: Random rotation, horizontal flipping, contrast adjustment and Cutmix

### Data Augmentation
To improve generalization, data augmentation is applied to the training images:

- **Random Rotation**: ±2.5% degrees
- **Random Flip**: Horizontal flip
- **Random Contrast**: ±10% variation in contrast

## Evaluation
The model is evaluated using standard classification metrics:

- **CategoricalAccuracy**
- **TopKCategoricalAccuracy**
- **Confusion Matrix**

### Sample Results:
The model achieves an accuracy of approximately 70% on the test set. Evaluation metrics are analyzed using a classification report and confusion matrix.

### Model Performance:
https://wandb.ai/PrakyathDl/Human_Emotion_Detection/reports/Human-Emotion-Detection-with-LeNet-Model-Performance-andEvaluation--Vmlldzo5OTk2MzU3

## Future Improvements
- **Model Architecture**: Experimenting with ResNet or other deeper models may improve performance.
- **Hyperparameter Tuning**: Adjusting dropout, regularization, and learning rate for better generalization.
- **Advanced Data Augmentation**: Adding techniques such as brightness and saturation adjustments to make the model more robust.

## Conclusion
This project demonstrates the use of CNNs for emotion detection from images. While LeNet provides a good starting point, there are opportunities to enhance performance with more complex architectures and optimized training techniques.

## References
- Kaggle Dataset: [Emotion Detection Images](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes)
