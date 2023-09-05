## Image-classification-project
<img src="https://pbs.twimg.com/media/Es71bGKUcAMKTem.png" width="400">


# Image Classification Project with CIFAR-10 Dataset and CNN Models
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

This project focuses on image classification using the CIFAR-10 dataset and employs different models such as SVM, Random Forest classifier, Logistic regression for multiclass problems and  Convolutional Neural Networks (CNNs). It was carried out on a standard laptop running Windows 11, utilizing different common python libraries plus TensorFlow and Keras libraries for CNN.
This repo contains 2 notebooks:

-**Image classification-SVM-RANDOM FOREST-LOGISTIC**: cifar 10 was loaded from keras and after an initial stage of simple data visualization it was necessary to process the data in accordance with their peculiarities (image data processing).
Subsequently a PCA was employed to reduce the complexity of the data and also to feed standard classification models correctly.
Each model was trained through the use of cross-validation and the main evaluation metrics are provided for each one (accuracies, mean and standard deviation, precision, recall and f1 score).
                      *SVC* 47%accuracy        *Random Forest classifier*  41% accuracy        *Logistic regression (multiclass)* 31%

-**Image classification-CNN**: The project begins with a thorough exploration of the CIFAR-10 dataset. After train-test split, Data visualization techniques are employed to gain insights into the dataset's characteristics. Additionally, data preprocessing is carried out to normalize the images, enhancing model performance plus one hot encoding for labels.
Cnn is the main model architecture used for the purpose of the project starting with a naive model (a lot of parameters but poor defined architecture) and then increasing its complexity by adding more layers and features (Convolutional, MaxPooling, Dropout, BatchNormalization).

The final model undergoes multiple training iterations and improvements were tracked using plots of accuracy and loss function (both for train and test), also, increasing its complexity helped solve overfitting problems.
Various metrics and visualization tools were used to assess model performance. These include accuracy, precision, recall, F1-score, and confusion matrices. These metrics help in understanding the strengths and weaknesses of the model.
                                                                *CNN* 76% accuracy


## Overview
<img src="https://miro.medium.com/v2/resize:fit:1182/1*OSvbuPLy0PSM2nZ62SbtlQ.png" width="200">


- **Dataset**: CIFAR-10, a popular dataset containing 60,000 32x32 color images in 10 different classes, with 6,000 images per class.
- **Objective**: To classify images into their respective categories using deep learning techniques and to fix some common problems involving machine learning.

## Project Highlights

1. **Data Visualization and Preprocessing**: This project begins with a comprehensive exploration of the CIFAR-10 dataset. This dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

After loading the dataset from Keras, i've performed an initial stage of simple data visualization to get a sense of the image data.
Data preprocessing was a critical step. It was necessary to process the data according to its peculiarities since it is image data and, according to the different models,this preprocess involved different techniques.

2. **Model Architectures**: For traditional machine learning approaches, the following models were used:
-Support Vector Classifier (SVC)
-Random Forest Classifier
-Logistic Regression (multiclass)
While, involving deep learning , Convolutional Neural Networks (CNNs) were the main focus. Starting with a naive CNN architecture and gradually increasing its complexity by adding layers and features such as Convolution, MaxPooling, Dropout, and BatchNormalization layers.
The final CNN architecture was designed to optimize performance and mitigate overfitting issues.

3. **Training Iterations**: Each model, whether traditional machine learning or deep learning, underwent multiple training iterations.
Cross-validation was employed to ensure robust model evaluation and performance assessment of SVM, RF and Log regression.
For the CNN model, training involved adjusting the architecture, batch and epochs size, multiple chunks of training (due to computational resources) and monitoring to achieve the best results.

4. **Model Accuracy and loss**:The accuracy of each model (and of course, minimization of loss) was a key focus throughout the project. The accuracy on both the training and test datasets to assess how well the models were learning and generalizing.
Here are the final accuracy results for each model:
-SVC: 47% accuracy
-Random Forest Classifier: 41% accuracy
-Logistic Regression (multiclass): 31%
-CNN: 76% accuracy

5. **Evaluation Metrics**: To gain a deeper understanding of model performance, various evaluation metrics beyond accuracy were employed. These metrics included:
-Precision
-Recall
-F1 score
Confusion matrices heatmaps: helping visualize and understand the strengths and weaknesses of each model

6. **Tools Used**: Throughout the project, several tools and libraries were employed to implement, train, and evaluate our models:
Common Python libraries for data manipulation and analysis.
TensorFlow and Keras libraries for deep learning model development.
Matplotlib and other visualization libraries for data visualization and plotting.
Scikit-Learn for traditional machine learning models like SVM, Random Forest, and Logistic Regression.






## Conclusion

In this project, we embarked on a journey of image classification using the CIFAR-10 dataset. Thr goal was to explore various model architectures, from traditional machine learning to deep learning, and to assess their performance through training iterations and evaluation metrics. What i've learned:

Data Preparation Matters:
It's crucial gaining a comprehensive understanding of the  dataset and proper data preprocessing plays a crucial role in enhancing model performance.

Model Diversity:
Experimenting with a range of model architectures, including Support Vector Classifier (SVC), Random Forest Classifier, Logistic Regression for multiclass problems, and Convolutional Neural Networks (CNNs) allowed me to compare the performance of traditional machine learning approaches with deep learning techniques.


Deep Learning Dominance in image classification:
The results, alongside modern scientific researches, clearly indicate that deep learning outperformed traditional machine learning models for the CIFAR-10 image classification task.
The CNN architecture, when properly designed and tuned, achieved higher level of accuracy, demonstrating its effectiveness in handling image data.

Training Iterations Pay Off:
Training iterations and cross-validation were instrumental in optimizing model performance.
Through defined processes and architectural adjustments, diferrent issues were fixed and the models' ability to generalize improved.

Evaluation Beyond Accuracy:
While accuracy is an important metric, it's not sufficient for a more comprehensive evaluation. 


In conclusion, this exploration of image classification on the CIFAR-10 dataset reinforces the power of deep learning, particularly CNNs, in handling complex visual data. 
However, it's important to  recognize the importance of careful data preprocessing, model architecture design, and training iterations to maximize performances.

This  findings and the lessons learned in this project can serve as a starting point for image classification and can be handled even with no such powerful computational resouces. 
Of course, there's room for major improvement but, at least, it can help handling common problems involving deep learning techniques.

Feel free to explore the accompanying Jupyter Notebook for a detailed walkthrough of the project's code and execution!
