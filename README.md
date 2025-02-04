# Predicting Applicants Loan Eligibility - Machine Learning

The project is divided into two phases. In the first phase (Phase 1), we conducted data cleaning and exploration. Afterwards, in Phase 2, the chosen machine learning models were applied to the cleaned dataset, and predictions were made based on the data.

## Dataset source

The data set that was used for this assignment is Loan Eligibility dataset. The data is taken from the Kaggle website (Devzohaib,2023). The data shows whether or not an applicant is granted for their loan application based on the information that each applicant provided.  
(Devzohaib.(2023). *Eligibility Prediction for Loan*. Kaggle. https://www.kaggle.com/datasets/devzohaib/eligibility-prediction-for-loan/data)

## Target Feature

The target feature from our dataset is Loan status variable. The Loan Status is a categorical feature where it is either yes or no. If `Yes`, this indicates that the individual was granted the home loan. If `No`, the individual's loan was rejected. This variable will be predicted based on the rest of variables that have been mentioned above.

## Goal and Objectives

Loan eligibility classification is crucial to bank which lead to the building of a prediction model that automatically classifies the loan status of applicants. The model will help the company to validate the applicants' eligibility and also improve the processing efficiency. In this case, the analysis aims to classify the loan eligibility for applicants based on their information they provided, such as married status, education, self employed status, total income, loan amount, credit history and many more features.

The objectives of the analysis are as following:
- Explore the relationship between the selected variables.
- Find which features that would be the best predictors of loan eligibility.
- Predict loan eligibility using machine learning models

## Methodology used

The following binary classifiers are used to predict the target feature, which is the Loan_Status: 
- K-Nearest Neighbors (KNN)
- Decision Tree (DT)
- Random Forest (RF)
- Naive Bayes (NB)
- SVC
- Neural Network (NN)

The modelling methodology begins by doing one hot encoding to the categorical feature into 0 or 1. After doing the one hot encoding, the transformation is done to minimise the effect of outliers and then the dataset is scaled using the min-max scaling. Furthermore, the full dataset is split into training and test set with a 70:30 ratio. This will give 414 rows as the training dataset and 178 rows as test dataset. Furthermore, prior fitting the algorithm to the training data, the best features are first selected. The feature selection is done using the powerful Random Forest Importance method where the 5 features that give the best score are used for the model fitting. 
Using the features that have been selected, the stratified K-Fold cross-validation is used before fitting the models so that equal proportions of the target feature are used for the training and testing datasets. The 5-fold stratified cross-validation is also conducted to fine-tune the hyperparameters of each classifier. We use the area under the curve (AUC) as the performance metric for all algorithms, except for neural networks (NN), which use accuracy as the performance metric. Each of the model is also build using parallel processing with "-2" cores. This is due to the target feature has an unbalanced target where the dataset has more observations for Loan Status is approved (Loan Status = Y). The results of each hyper-parameter tuning for each classifiers is then assessed and visualised using plots. Furthermore, for the NB classifier, the PowerTransformer method is applied to the training data due to GaussianNB method assume normal distribution. 

The tuned classifiers are the classifiers that have been optimised using the best hyper-parameter values that are identified through grid search. Once all six tuned classifiers (with the best hyper-parameter values) are identified, the tuned classifiers are then fitted into the test data using 5-fold cross validation. Lastly, the paired t-tests are then conducted to see if there are any performances that differ significantly based on the statistic test. 

## Summary of findings

In this case, we leverage Random Forest Importance (RFI) to identify key features that verify loan applicant eligibility, thereby improving a company's processing efficiency. By training a random forest model on the dataset, we extracted feature importance and selected the most relevant features. Our model shows that the first five features: `Self_Employed`, `CoapplicantIncome`, `Loan_Amount_Term`, `Credit_History`, and `ApplicantIncome`, are critical in determining an applicant's eligibility.

Furthermore, The Random Forest (RF) model with 5 features selected by Random Forest Importance (RFI) performs the best with the highest cross-validated AUC score of 0.82 on the training data. Additionally, there are significant differences between the AUC score of Random Forest (RF) model and Naive Bayes (NB) model, K-Nearest Neighbors (KNN) model, Decision Tree (DT) model, and SVC model due to their p-value less than 0.05. On the contrary, there is no significant difference between the AUC score of Random Forest (RF) model and Neural network (NN) model because its p-value is greater than 0.05. Therefore, RF and NN models could be used to predict the loan eligibility in this case. However, when comparing the computational cost of these two models, RF model is a better choice for the prediction of loan eligibility.

## Conclusions

After the analysis of all the classifier models, the goals and objectives of our project has been achieved. The random forest model has been chosen as the best model to help the company to improve the processing efficiency of the validation process of the applicants' eligibility with less resource usage. By choosing only some of the best features such as `Self_Employed`, `CoapplicantIncome`, `Loan_Amount_Term`, `Credit_History`, and `ApplicantIncome` to predict the applicants' eligibility, it can also help the company improve the overall operational efficiency and decision accuracy.

In summary, the application of the random forest model and the identification of the top five features would greatly improve  the company's loan eligibility processing process, providing a more reliable and efficient method for assessing applicant eligibility while optimizing resource utilization.
