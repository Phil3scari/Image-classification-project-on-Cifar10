## Image-classification-project
<img src="https://pbs.twimg.com/media/Es71bGKUcAMKTem.png" width="400">


# CIFAR10 Image classification
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

This project focuses on image classification using the CIFAR-10 dataset and employs different models such as SVM, Random Forest classifier, Logistic regression for multiclass problems and  Convolutional Neural Networks (CNNs). 
It was carried out on a standard laptop running Windows 11, Jupyter notebook and common Python libraries plus TensorFlow-Keras for Deep Learning.

## Overview
<img src="https://miro.medium.com/v2/resize:fit:1182/1*OSvbuPLy0PSM2nZ62SbtlQ.png" width="200">

This repo contains 2 notebooks:
-**Image classification-SVM-RANDOM FOREST-LOGISTIC**: Classification with traditional Machine Learning.
Involves Data visualization, data preprocessing, PCA, train-test, model selection, model assessment and comparison through different score metrics.
                      
-**Image classification-CNN**: Deep Learning classification.
Involves Data visualization, data preprocessing (+encoding), increasing model complexity, train-test and metrics to assess improvements over time.


## Project Highlights

1. **Data Visualization and Preprocessing**: This project begins with a comprehensive exploration of the CIFAR-10 dataset.
This dataset consists of 60.000 32x32 color images in 10 different classes, with 6.000 images per class.
After loading the dataset from Keras, I've performed an initial stage of simple data visualization to get a sense of the image data.
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

4. **Model Accuracy and loss**:The Accuracy of each model (and of course Loss) was a key focus throughout the project. 
Here are the final accuracy results for each model:
-SVC: 47% accuracy
-Random Forest Classifier: 41% accuracy
-Logistic Regression (multiclass): 31%
-CNN: 76% accuracy

5. **Evaluation Metrics**: To gain a deeper understanding of model performance, various evaluation metrics beyond accuracy were employed. These metrics included:
-Precision, Recall, F1-Score
-Confusion matrices heatmaps: helping visualize and understand the strengths and weaknesses of each model

6. **Tools Used**: Throughout the project, several tools and libraries were employed to process data and implement, train, and evaluate our models:
Common Python libraries for data manipulation and analysis.
TensorFlow and Keras libraries for deep learning model development.
Matplotlib and seaborn for data visualization and plotting.
Scikit-Learn for traditional machine learning models like SVM, Random Forest, and Logistic Regression.



## Conclusion (what I've learned)

**Data Preparation Matters**:
It's crucial gaining a comprehensive understanding of the  dataset and proper data preprocessing plays a crucial role in enhancing model performance.

**Model Diversity**:
Experimenting with a range of model architectures, including Support Vector Classifier (SVC), Random Forest Classifier, Logistic Regression for multiclass problems, and Convolutional Neural Networks (CNNs) allowed me to compare the performance of traditional machine learning approaches with deep learning techniques.

**Deep Learning Dominance in image classification**:
The results, alongside modern scientific researches, clearly indicate that deep learning outperformed traditional machine learning models for the CIFAR-10 image classification task.
The CNN architecture, when properly designed and tuned, achieved higher level of accuracy, demonstrating its effectiveness in handling image data.

**Training Iterations Pay Off**:
Training iterations and cross-validation were instrumental in optimizing model performance.
Through defined processes and architectural adjustments, different issues were fixed and the models' ability to generalize improved.

**Evaluation Beyond Accuracy**:
While accuracy is an important metric, it's not sufficient for a more comprehensive evaluation. 


In conclusion, this project reinforces the power of deep learning, particularly CNNs, in handling complex visual data. 
However, it's important to  recognize the importance of careful data preprocessing, model architecture design, and training iterations to maximize performances.
This  findings and the lessons learned in this project can serve as a starting point for image classification and can be handled even with no such powerful computational resouces. 
Of course, there's room for improvements.

Feel free to explore the accompanying Jupyter Notebooks for a detailed walkthrough of the project's code and execution!


## External References
[1] https://scikit-learn.org/stable/user_guide.html
[2] https://keras.io/guides/
[3] https://www.tensorflow.org/tutorials/images?hl=it
