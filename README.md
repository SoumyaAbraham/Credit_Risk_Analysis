# Credit_Risk_Analysis

Credit Risk sees good loans outnumbering risky ones. Therefore, it is important to employ various training and evaluation techniques so that the model can get a good understanding of the data.

### DELIVERABLE 1

In this project, we will be using *imbalanced-learn* and *scikit-learn* libraries to build and evaluate models using resampling.

We will evaluate three machine learning models and determine which is the best for predicting credit risk. 
You can find the code for this part of the project [here](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb)  

The steps involved in this analysis are as follows:  

Before we start, import all the dependencies for this project.

STEP 1: Transform the data into a usable form which involves:

  * Loading the data
  * Dropping _NULL_ values from colummns and rows
  * Converting strings to numerical datatypes
  * Converting target column values to _High Risk_ and _Low Risk_ based on their values
  
  ![transform](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del1-%20load.PNG)
  
STEP 2: Split the data into Training and Testing sets

  ![split](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del1-%20Split.PNG)
  
  Going a little further, we can  
  * Check the balance of target values
  * Check the shape of the X training set
  
  ![balance and shape](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del1-%20Balance%20and%20Shape.PNG)

STEP 3: Oversampling: Here you will compare two oversampling algorithms to determine which perfomrs better.

  * Using Naive Random Oversampling
  
  ![Naive Oversampling](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del1-%20Naive%20Oversampling.PNG)
  
  * Using SMOTE Oversampling
  
  ![SMOTE Oversampling](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del1-%20SMOTE%20Oversampling.PNG)
  
STEP 4: Undersampling: Let us use Cluster Centroids Algorithm here. 
  
  ![Undersampling](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del1-Undersampling.PNG)
  
  
### DELIVERABLE 2
  
STEP 5: Over and Under Sampling (SMOTEENN)

  ![SMOTEENN](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del2-SMOTEENN.PNG)
  
### DELIVERABLE 3

Here, we will use _imblearn.ensemble_, _BalancedRandomForestClassifier and _EasyEnsembleClassifier_ to predict credit risk and evaluate each model.

You can find the code for this part of the project [here](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)  

Before we start, ensure you have installed all the necessary libraries. If not, do a quick pip install imblearn and pip install -U scikit-learn. Bring in all the dependencies as well.

STEP 1: Much like the before, bring in the CSV and clean it up so it can be used for risk analysis and testing.

STEP 2: Split the data into Training and Testing sets

  ![split](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del3-Split1.PNG)
  ![split2](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del3-Split2.PNG)
  
STEP 3: Ensemble Learners: Here, you will train a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier to see which one gives better results.

  * Balanced Random Forest Classifier:
  
  ![BRFC](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del3-BRFC.PNG)
  
  * List the features sorted in descending order by feature importance
  
  ![Importance](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/del3-importance.PNG)
  
  * Easy Ensemble AdaBoost Classifier
  
  ![AdaBoost](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/Del3-AdaBoost.PNG)
  
### ANALYSIS

Let us compare the various results:

Naive Oversampling Results

![NOS](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/naiveOversamplingAnalysis.PNG)

SMOTE Results

![SMOTE](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/SMOTE%20Analysis.PNG)

Cluster Centroid Results

![Cluster](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/ClusterCentroidAnalysis.PNG)

SMOTEENN Results

![SMOTEENN](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/SMOTEENN%20Analysis.PNG)

Easy Ensemble Results

![ADABOOST](https://github.com/SoumyaAbraham/Credit_Risk_Analysis/blob/main/Images/del3Analysis.PNG)

From these results, we notice very low precisions for High_Risk factor. This is an indication of a large number of False Positives.

#### When we look at EasyEnsembleClassifier model, we notice it has higher scores for High Risk loans and the balance accuracy is much higher than that of any other model. 
#### Therefore, it can be concluded that EasyEnsembleClassifier models are the most effective of them all.
