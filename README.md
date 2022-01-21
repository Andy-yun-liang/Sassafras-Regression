# Sassafras-Regression


## Table of Contents
   
* [Background](#background)
* [Model Summary](#model-summary)
* [Data Preprocessing](#data-preprocessing)
* [Model Tuning](#model-tuning)


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
| Linear Stack (GBM + XGB + LGBM)|8.561| 8.501|


## Data Preprocessing
I started off my data preprocessing phase by changing the columns with the “?” strings into a missing value. Next, I applied a simple mean fill for the missing values as imputation methods such as K nearest neighbours are too memory-heavy for the given dataset even with the free computing power provided . Next, I removed features that were under an arbitrary variance threshold because low variance features don’t contribute a lot to a model’s predictive ability. I started off with a threshold of 0 and ended up choosing a threshold of 0.05 for my final training and testing dataset as it resulted in better RMSE overall when comparing the models. After removing the low variance features, I decided to remove features with a correlation of the absolute value of 0.8 to reduce the training time. A justification for this choice is that linear-based methods will improve. Finally, I scaled the data with the standard scaler function in case I decide to use algorithms that are magnitude dependent. 

### Changing ? to NaNs
```python
trainX["#B17"] = trainX["#B17"].replace("?",np.nan).astype('float64') 
testX["#B17"] = testX["#B17"].replace("?",np.nan).astype('float64') 
```

### Mean Fill
```python
trainX = trainX.fillna(trainX.mean())
testX = testX.fillna(testX.mean())
```

### Removing Low Variance Features
```python
from sklearn.feature_selection import VarianceThreshold
the_threshold = VarianceThreshold(threshold=0.05)
the_threshold.fit(trainX)
to_be_remove = [i for i in testX.columns
               if i not in testX.columns[the_threshold.get_support()]]
testX.drop(to_be_remove,axis=1,inplace=True)
trainX.drop(to_be_remove,axis=1,inplace=True)
```

### Removing Highly Correlated Features
```python
corr = trainX.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
#from sklearn feature selection documentation
def correlation(dataset, threshold):
    col_corr = set()  
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  
                col_corr.add(colname)
    return col_corr
corr_ft = correlation(trainX, 0.8)
trainX.drop(corr_ft,axis=1,inplace=True)
testX.drop(corr_ft,axis=1,inplace=True)
```

## Model Tuning
For the Model Tuning process, I decided to use the scikit-learn library with gridsearchcv and randomsearchcv as it's the most extensive library for machine learning in Python asides from Tensorflow and Pytorch.

Setting up the cross validation evaluation
```python
def the_rmse(model):
    kf = KFold(3, shuffle=True, random_state=12).get_n_splits(trainX3)
    rmse= np.sqrt(-cross_val_score(model, trainX3, trainY_final, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
```

### Linear model
```python
ls=LinearRegression()
the_rmse(ls).mean()
```

### Lasso model
```python
lasso_alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
lasso_rmse = []

for i in lasso_alpha:
    lasso = make_pipeline(RobustScaler(), Lasso(alpha = i,random_state = 12))
    lasso_rmse.append(the_rmse(lasso).mean())

lasso_rmse_results = pd.DataFrame(lasso_rmse,lasso_alpha,columns=['RMSE'])
display(lasso_rmse_results.transpose())

```

### Ridge model
```python
ridge_alpha = list(range(1,16))
ridge_rmse = []

for val in ridge_alpha:
    ridge = make_pipeline(RobustScaler(), Ridge(alpha = val,random_state = 12))
    ridge_rmse.append(the_rmse(ridge).mean())
ridge_rmse_results = pd.DataFrame(ridge_rmse,ridge_alpha,columns=['RMSE'])
display(ridge_rmse_results.transpose())
```

### ElasticNet 
```python
enet_alpha = [0.001,0.01,0.1,0.25,0.5,0.75,1]
enet_ratio = [0.001,0.01,0.03,0.05,0.1,0.15,0.25,0.4]
enet_rmse = []

for a in enet_alpha:
    for r in enet_ratio:
        enet = make_pipeline(RobustScaler(), ElasticNet(alpha = a,l1_ratio=r,random_state = 12))
        enet_rmse.append(the_rmse(enet).mean())
```

### Regression Tree
```python
param_grid = {'max_depth' : [4,5,6,7,8,9,10,15] ,
              'max_features' : [4,5,6,7,8,9,10,15]
             }
tree_mod = DecisionTreeRegressor(random_state=12)
tree_grid = GridSearchCV(tree_mod, param_grid, cv=10, refit=True, verbose=1, scoring = 'neg_mean_squared_error')
tree_grid.fit(trainX,trainY_final)
```

### Random Forest
```python
rf_params = {
 'max_features':[5,10,15,20,25,35,40,45]
 }

rf_grid = GridSearchCV(RandomForestRegressor(random_state=12), rf_params,refit=True, verbose=1, scoring ='neg_mean_squared_error')
rf_grid.fit(trainX,trainY_final)
```

### Gradient Boosting Machine
```Python
gbm_params ={'learning_rate' : [0.05, 0.1, 0.3, 0.5, 0.6, 0.75, 0.9, 1.1, 1.2],
            'n_estimators' : [800, 1200, 1500],
            'max_depth' : [3, 5, 7, 9, 12],
            'max_features' : [5, 8, 10, 12, 15]}

gbm_mod = GridSearchCV(GradientBoostingRegressor(random_state=12), gbm_params, scoring = 'neg_mean_squared_error', verbose = 1)
gbm_mod.fit(trainX,trainY_final)

```

### Extreme Gradient Boost
```python
xgb_params = {'n_estimators' : [500, 700, 1000],
              'max_depth' : [3, 4, 5, 7],
              'learning_rate' : [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
              'reg_lambda' : [1.3, 1.5, 1.6, 1.7]
              }

xgb_mod = GridSearchCV(XGBRegressor(metric="rmse"), xgb_params, scoring = 'neg_mean_squared_error', verbose = 1, cv = 3)
xgb_mod.fit(trainX, trainY_final)
```

### Light Gradient Boost
```python
lgbm_params = {'n_estimators' : [500, 700],
               'max_depth' : [3, 5, 7, 9],
               'reg_lambda_l2' : [1, 1.2, 1.6]
               'num_leaves' : [20, 30, 40, 50, 70]
               'learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ]}

lgbm_mod = GridSearchCV(LGBMRegressor(objective = 'regression', metric = 'rmse'), lgbm_params, scoring = 'neg_mean_squared_error',verbose=1, cv = 10)
lgbm_mod.fit(trainX,trainY_final)
```
### Average Ensemble
```python
the_weights = [1/3, 1/3, 1/3]
preds['prediction'] = (predictions_data[0]*the_weights[0]) + (predictions_data[1]*the_weights[1]) + (predictions_data[2]*the_weights[2])
```

### Weighted Average Ensemble 
With more weight distributed to XGBoost as it's a better stand alone regressor compared to GBM and LGBM.
```python
weighted_weights = [0.6,0.2,0.2]
preds['prediction'] = (predictions_data[0]*weighted_weights[0]) + (predictions_data[1]*weighted_weights[1]) + (predictions_data[2]*weighted_weights[2])
```

### Random Forest Stack
```python
random_forest_mod = RandomForestRegressor(max_depth = 45,bootstrap=True)
the_regressors = StackingRegressor(regressors = [xgb_final_mod,lgbm_final_mod,gbm_mod], meta_regressor = random_forest_mod)
the_regressors.fit(trainX3,trainY_final)
stacked_preds = the_regressors.predict(testX3)
```

### Linear Stack
```python
lm_mod = LinearRegression()
the_regressors2 = StackingRegressor(regressors = [xgb_final_mod,lgbm_final_mod,gbm_mod], meta_regressor = lm_mod)
the_regressors2.fit(trainX3,trainY_final)
stacked_preds2 = the_regressors2.predict(testX3)

```
