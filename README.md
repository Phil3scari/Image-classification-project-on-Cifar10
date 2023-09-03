## Image-classification-project
<img src="https://pbs.twimg.com/media/Es71bGKUcAMKTem.png" width="400">


# Image Classification Project with CIFAR-10 Dataset and CNN Models
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

This project focuses on image classification using the CIFAR-10 dataset and employs Convolutional Neural Networks (CNNs) as model architecture. It was carried out on a standard laptop running Windows 11, utilizing TensorFlow and Keras libraries.

## Overview
<img src="https://miro.medium.com/v2/resize:fit:1182/1*OSvbuPLy0PSM2nZ62SbtlQ.png" width="200">


- **Dataset**: CIFAR-10, a popular dataset containing 60,000 32x32 color images in 10 different classes, with 6,000 images per class.
- **Objective**: To classify images into their respective categories using deep learning techniques and to fix some common problems involving machine learning.

## Project Highlights

1. **Data Visualization and Preprocessing**: The project begins with a thorough exploration of the CIFAR-10 dataset. After train-test split, Data visualization techniques are employed to gain insights into the dataset's characteristics. Additionally, data preprocessing is carried out to normalize the images, enhancing model performance plus one hot encoding for labels.

2. **Model Architectures**: The project starts with a naive CNN  (with a lot of parameters but with a poor architecture) and progressively increases layers complexity . It experiments with convolutional layers,MaxPooling layers, dropout layers and batch normalization layers.

3. **Training Iterations**: The final model undergoes multiple training iterations to fine-tune its performance. Hyperparameters, such as learning rates, are progressively lowered.

4. **Model Accuracy**: After training and optimization, the project achieves an accuracy rate of 76% on the CIFAR-10 dataset handling overfitting.

5. **Evaluation Metrics**: Various metrics and visualization tools are used to assess model performance. These include accuracy, precision, recall, F1-score, and confusion matrices. These metrics help in understanding the strengths and weaknesses of the model.

6. **Tools Used**: The project is implemented using Jupyter Notebook, TensorFlow, and Keras on a Windows 11 laptop.

## Conclusion

This image classification project showcases the power of CNNs in classifying images from the CIFAR-10 dataset. Through careful data preprocessing, model architecture design, and hyperparameter tuning, the project successfully achieves a 76% accuracy rate. The evaluation metrics provide insights into the model's performance and its ability to classify images across multiple categories.

Of course, there's room for major improvement but, at least, it can help handling common problems involving deep learning techniques.

Feel free to explore the accompanying Jupyter Notebook for a detailed walkthrough of the project's code and execution!


