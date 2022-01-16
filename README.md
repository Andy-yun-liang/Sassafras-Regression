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
| ElasticNet| 10.637|10.238| 
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
For the Model Tuning process, I decided to use the scikit-learn library with gridsearchcv and randomsearchcv as it's the most extensive library for machine learning in Python asides from Tensorflow and Pytorch.

Setting up the cross validation evaluation
```python
def the_rmse(model):
    kf = KFold(3, shuffle=True, random_state=12).get_n_splits(trainX3)
    rmse= np.sqrt(-cross_val_score(model, trainX3, trainY_final, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
```

# Linear model
```python
ls=LinearRegression()
the_rmse(ls).mean()
```


# Lasso model
```python
lasso_alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
lasso_rmse = []

for i in lasso_alpha:
    lasso = make_pipeline(RobustScaler(), Lasso(alpha = i,random_state = 12))
    lasso_rmse.append(the_rmse(lasso).mean())

lasso_rmse_results = pd.DataFrame(lasso_rmse,lasso_alpha,columns=['RMSE'])
display(lasso_rmse_results.transpose())

```

# Ridge model
```python
ridge_alpha = list(range(1,16))
ridge_rmse = []

for val in ridge_alpha:
    ridge = make_pipeline(RobustScaler(), Ridge(alpha = val,random_state = 12))
    ridge_rmse.append(the_rmse(ridge).mean())
ridge_rmse_results = pd.DataFrame(ridge_rmse,ridge_alpha,columns=['RMSE'])
display(ridge_rmse_results.transpose())
```

# ElasticNet 
```python
#enet
enet_alpha = [0.001,0.01,0.1,0.25,0.5,0.75,1]
enet_ratio = [0.001,0.01,0.03,0.05,0.1,0.15,0.25,0.4]
enet_rmse = []

for a in enet_alpha:
    for r in enet_ratio:
        enet = make_pipeline(RobustScaler(), ElasticNet(alpha = a,l1_ratio=r,random_state = 12))
        enet_rmse.append(the_rmse(enet).mean())
```




## Final Model
