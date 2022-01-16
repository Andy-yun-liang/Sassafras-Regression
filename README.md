# Sassafras-Regression


## Table of Contents
   
* [Background](#background)
* [Model Summary](#model-summary)
* [Data Preprocessing](#data-preprocessing)
* [Model Tuning](#model-tuning)
* [Final Model](#final-model)


## Background
This is my regression analysis of the Sassafras regression competition for STAT 440 Learning from Big Data.

The dataset consist of over 200,000 randomly simulated observations with 75 covariates, more information about the dataset can be found at https://www.kaggle.com/c/stat440-21-project3b/data.


## Model Summary

| Model | Private RMSE | Public RMSE |
| :---  | :---:    |  :---:  | 
| Multiple Regression   | 12.365 | 11.389 | 
| Lasso | 10.644 |  10.412|
| Ridge | 10.629 |10.107|
| Elasticnet| 10.637|10.238| 
| Regression Tree| 9.816| 9.709|
| Random Forest| 8.946| 8.834|
| GBM | 8.855| 8.713|
| XGBoost | 8.561| 8.435|
| LightGBM| 8.681| 8.522|
| AVG ENSEMBLE (GBM + XGB + LGBM) |8.521|8.371|
| W.AVG ENSEMBLE (GBM + XGB + LGBM)|8.525|8.377|
| RF Stack (GBM + XGB + LGBM)|8.831|8.742|
| Lasso Stack (GBM + XGB + LGBM)|8.561| 8.501|


## Data Preprocessing
I started off my data preprocessing phase by changing the columns with the “?” strings into a missing value. Next, I applied a mean fill for the missing values as imputation methods such as K nearest neighbours are too memory-heavy for the given dataset. Next, I removed features that were under an arbitrary variance threshold because low variance features don’t contribute a lot to a model’s predictive ability. I started off with a threshold of 0 and ended up choosing a threshold of 0.05 for my final training and testing dataset as it resulted in better RMSE overall when comparing the models. After removing the low variance features, I decided to remove features with a correlation of the absolute value of 0.7 to reduce the training time. A justification for this choice is that linear-based methods will improve. Finally, I scaled the data with the standard scaler function in case I decide to use algorithms that are magnitude dependent. 


## Model Tuning


## Final Model
