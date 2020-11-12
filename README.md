# Machine-learning-Kaggle-competition

# Introduction

The competition consisted in estimating predictions of a binary classification problem proposed in the Kaggle platform. The database provided included several problems such as imbalance, missing values, linear and nonlinear combination between variables, as well as noisy features.
The evaluation method for the predictions of the test set was the F1 score, which is based in the precision and recall statistics, relating the true positives with the false positives and false negatives, respectively.
In the following sections, we explain several pre-processing steps that we performed to solve the initial database problems, as well as multiple methods to try to find an optimal F1 score.

# Pre-processing

We performed several steps in the pre-processing. The first problem we had to face was the missing values of the dataset. After some research, we decided to perform imputation through the Multiple Imputation by Chained Equations (MICE) method. This method was computationally very costly, but it performed better than a simple imputer.
Afterwards, we dealt with the multicollinearity problem of the dataset. We observed several variables that were very highly correlated between them and created a method to eliminate those which had a correlation higher than 0.7. From the initial 147 features, we were left with 124.

In order to test our methods, we partitioned the training dataset into train and validation using a random sampler, under an 80/20 proportion. Since the dataset was imbalanced in terms of the target feature, we performed both under sampling and oversampling to treat the problem and help the methods to perform correctly.

We used several techniques for this last step, like random under sampling, random oversampling or the Synthetic Minority Oversampling Technique (SMOTE). The one that produced better results under several methods was under sampling, reducing the train split so that half of the observations have a target equal to 0 and the other half equal to 1.
The methods were trained in this balanced train split and tested in the validation split, in order to select the best models to test using the entire training set, producing the target results of the testing set. One last step was done before trying the methods, which was to standardize the variables.

# Models

Several methods have been used to perform the binary classification. For some of these methods, the tuning of the hyper-parameters has been carried out and for others not, due either to the high execution time that this entailed or because a particular method did not support tuning.
The F1 score metric was used to evaluate each method, comparing the target of the validation part with the predicted target by the method in question. Once the F1 score was calculated, it was be divided into two values, one for each class, looking at the F1 score for class 1 as a measure to optimize. For most methods, the sklearn library was used, except for XGBClassifier, where xgboost was used.

We trained the data using the following methods:
    • Logistic Regression (LR): tuning of the hyper parameter C was performed, among which various values of said parameter were explored with an exhaustive search using the           gridsearchCV method. The number of folds that was specified was 20 since the time of this method is not very high and F1 was used as scoring, since it is the metric used         in the competition.
    • Decision Tree classifier (DT), K-Nearest Neighbours’ classifier (KNN) and Random Forest classifier (RF): in this case the tuning of the hyper parameters was not                 performed due to the high execution time that this required.
    • Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA) and Gaussian Naïve Bayes (GNB) do not have parameters to perform tuning.
    • XGBoost classifier (XGB): several hyper parameters were tuned, such as the number of estimators, the number of subsamples, minimum child weight, maximum depth, learning         rate, game and col sample bytree.
    • Keras Neural Network (Keras): a neural network was built with Keras, establishing various dense layers with a relu activation function, to end with a last dense layer with       sigmoid as activation function. Furthermore, the binary cross entropy was used as the functions, Adam as optimizer, and F1 as the metric. In addition, various numbers of         epochs and batch sizes were tested in order to increase the accuracy of the metric used.
    • Stacking methods (Stack): an attempt was made stacking various methods described above, in order to combine their predictions and obtain a better result. A logistic             regression was used as the final estimator.
    
# Results    

The following table shows the maximum results for the F1 score obtained in the validation partition under each of the methods used:

| Method   | LR     | DT     | LDA    | QDA    | RF     | KNN    | GNB    | XGB    | Keras  | Stack  |
|----------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| F1 score | 0.2899 | 0.2017 | 0.2894 | 0.2388 | 0.2799 | 0.1999 | 0.3091 | 0.3268 | 0.2432 | 0.3205 |


# Conslusions

As we can see in the previous table, the greatest F1 score is obtained with the XGBoost classifier, obtaining a maximum score of 0.3268 in the validation part. We obtained the best results under this method and, thus, this was the most used method to submit predictions with.
Once we loaded our results on the Kaggle platform, we obtained a final score of 0.3192, quite like what we achieved in the validation partition.
