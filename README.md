#  GREEN CHEMISTRY: PREDICTING SOLVENT 'GREENNESS' USING ML METHODS 

# CONTENTS: 
## 1. Introduction
## 2. Correlations
## 3. Training
   ### - Classification models 
   ### - Regression models
## 4. Testing 
   ### - SHE
   ### - Classification

# -------------------------------------------------------------------------
## 1. Introduction

Four classification and 18 regression models were built using scikit-learn. 5 datasets
with solvents used in industry were developed. The classification dataset (Dataset.csv) includes 52 solvents with 11 
descriptors. Safety (Safety.csv), health (Health.csv), and environmental (Environmental.csv) datasets were developed for
the regression models. 
The safety model uses 6 descriptors, health uses 4 and environmental uses 5. 
Descriptors were plotted against each other to identify any correlations. 
A final dataset (NewDataset.csv) was developed to test the developed models on. 

## -----------------------------------------------------------------------------------------

## 2. Correlations

Import dataset then drop the name and dependent variable columns. 
Define a variable with all the other columns in the dataset. 
Creating a function to impute the mean and fit to the descriptors. 
Scale and transform the data then create a dataframe. 
Define a variable for all possible combinations of descriptors to plot against each other.
Create a list of figures then write each image with a specific file directory for each combination.

## -----------------------------------------------------------------------------------------

## 3. Training

### Classification:

#### FILES: classification.py, ML.py

##### 1. classification.py

###### Parameters: Descriptors used in the dataset, machine learning method, and cross-validation method. 

Import the complete dataset as a csv file. 

 Define the descriptors.

 Set y to the dependent variable (EG: Sustainability score).

 Impute the mean for missing values, fit, and transform the data. 

 Scale the data and then transform again.

 Use the transformed dataset to create a dataframe, giving the now complete dataset with no missing values.

 Define the method you wish to use.

 Define the cross validation train:test splits you wish to use. 

 Create lists for the expected scores and the predictions made.

 Define what data is the testing and the training set.

 Remove the column for the name of solvents. 

 Fit the model and add the prediction and expected score to the lists.

 Print a confusion matrix and classification report. 

 There's also a section on optimising hyperparameters. Just need to define the param grid with 
certain parameters in the method, the values depend on the method/hyperparameter. 
After just set up a grid search cross validation and fit your model to it.

##### 2. ML.py

 Import the classification file and classification function. 

Define the descriptors you wish to use in the classification model. The names must be identical to those
used in the dataset. 

Define the variables you want to get and call the classification function. 

-----------------------------------------------------------------------------------------------

### Regression:
#### FILES: Safety.py, Health.py, Environmental.py


##### a. Importance_method function (all three files):

###### Parameters: list of feature importance and descriptors used in RF model. 

Create a bar graph showing the average importance of each feature in each split for Random Forest.
Error bars are the standard deviation for the dataset. 

##### b. Safety, health, and environmental score functions 

###### Parameters: Descriptors used in the model, machine learning method, cross-validation method

Import the safety dataset with the descriptors you have chosen for the model.
Create a variable to define those descriptors. 
Set y to be the dependent variable. 
Create a function to impute the mean for missing values. 
Fit that impute function to the descriptors variable. 
Scale and transform the data then create a dataframe. 

Define the methods and cross validation splits you want to use. 
Create lists for the predictions, real scores and feature importance. 
Define the training and testing sets. 
For analysis of multiple linear regression the coefficients and intercepts can be taken.
For random forest, easiest to add the feature importance for that model to the list and print that list. 

Round the predictions to the nearest whole number.
Retrieve r squared and mean absolute error values. 
Call the importance_method function if using random forest. 

Define the descriptors you want to use and call the function. 

## -----------------------------------------------------------------------------------------

## 4. Testing

#### FILES: Safety_score.py, Health_score.py, Environmental_score.py, TestDataset.py, Combination.py
#### DATASETS: Safety_New_Dataset.csv, Health_New_Dataset.csv, Environmental_New_Dataset.csv, NewDataset.csv

The same descriptors were used in the new datasets as the original datasets. 

##### Test_model function: 

###### Parameters: training dataset, test dataset, descriptors, and dependent variable. 

Define the X train/test and Y train/test splits. 
Function for imputing the mean for missing values. 
Fit the X train data to this impute function. 
Scale and transform the X train data. 
Define the model and fit it to the X train and y train.
Print a list of the predictions made rounded to the nearest whole number. 

Define the training data as the original dataset.

Define the testing data as the new dataset.

Define the descriptors.

Call the function.

---------------------------------------------------------------------------------------------















