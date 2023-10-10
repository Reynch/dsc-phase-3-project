# SyriaTel Churn Predictions

# Overview
![Towers]((https://st.depositphotos.com/1968353/2536/i/450/depositphotos_25360787-stock-photo-communication-towers.jpg))

Our dataset consists of 3333 entries from customers with different phone plans, usage rates of different services, and customer service calls as well as if they churned or not.

# Bussiness Understanding
We would like to use a Logistic Regression model to help predict which customers will churn so that we can identify and try to prevent churn.


```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer,  make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix, recall_score,\
    accuracy_score, precision_score, f1_score, plot_roc_curve, classification_report

from sklearn.dummy import DummyClassifier, DummyRegressor

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImPipeline

import warnings
warnings.filterwarnings('ignore')
```
# Data



## Importing Data

```python
churn_data = pd.read_csv('./data/bigml_59c28831336c6604c800002a.csv')

churn_data.head()

churn_data.info()

churn_data['churn'].value_counts(normalize = True)
```

The data is imbalanced marginally but could potentially use some balancing when we do our predictions

## Train Test Split
```python
X = churn_data.drop(['phone number','churn'], axis = 1)
y = churn_data.churn

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)
```


# Creating Pipelines with simpleimputer to prepare for data which may have null values in future
```python
subpipe_numeric = Pipeline(steps =[
    ('num_impute', SimpleImputer()),
    ('ss', StandardScaler())
])

subpipe_cat = Pipeline(steps = [
    ('cat_impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(sparse = False, handle_unknown='ignore'))
])
```
# Basic Colum Transformer
```python
CT = ColumnTransformer(transformers = [
    ('subpipe_num', subpipe_numeric,numeric_cols),
    ('subpipe_cat', subpipe_cat, cat_cols)
])
```
# Models

## Baseline Model
```python
dummy_model_pipe = Pipeline([
    ('ct',CT),
    ('dummy', DummyClassifier(strategy = 'most_frequent'))
])

dummy_model_pipe.fit(X_train,y_train)

dummy_model_pipe.score(X_train, y_train)

dummy_model_pipe.score(X_test, y_test)

print(classification_report(y_test, dummy_model_pipe.predict(X_test)))
```
Baseline model predicting majority class has an accuracy of 86%, I would like to maximize recall without sacrificing accuracy in an attempt to retain as many customers as possible without having to offer coupons or discounts unnecessarily 

# Looking to see if there are columns that may improve accuracy or recall based on correlation
```python
fig, ax = plt.subplots(figsize = (12,8))
sns.heatmap(churn_data.drop(['state', 'international plan', 'voice mail plan'], axis = 1).corr(), annot = True)
```
## Basic Logistic Regression Model
```python
logreg = Pipeline(steps = [
    ('ct', CT),
    ('logreg', LogisticRegression())
])

logreg.fit(X_train, y_train)

logreg.score(X_train, y_train)

logreg.score(X_test, y_test)

plot_roc_curve(logreg, X_test, y_test)

plot_confusion_matrix(logreg, X_test, y_test)

print(classification_report(y_test, logreg.predict(X_test)))
```
Not a great recall score, I think we should err on the side of caution and try to identify as many cases as churn as possible without drastically decreasing our accuracy score to prevent as many people as possible from leaving the service by perhaps offering incentives on the positively correlated items that seem to have people leave the service

## No states Model
```python
cat_cols_no_states = cat_cols[1:]

CT_no_states = ColumnTransformer(transformers = [
    ('subpipe_num', subpipe_numeric,numeric_cols),
    ('subpipe_cat', subpipe_cat, cat_cols_no_states)
])

logreg_no_states = Pipeline([
    ('ct', CT_no_states),
    ('logreg', LogisticRegression())
])

logreg_no_states.fit(X_train, y_train)

plot_roc_curve(logreg_no_states, X_test, y_test)

plot_confusion_matrix(logreg_no_states, X_test, y_test)

print(classification_report(y_test, logreg_no_states.predict(X_test)))
```
Looks like our recall decreased and our accuracy only marginally decreased. I think I will keep the states in the model as it seems to help a little bit.

## Correlated columns Model
```python
fig, ax = plt.subplots(figsize = (12,8))
sns.heatmap(churn_data.drop(['state', 'international plan', 'voice mail plan'], axis = 1).corr(), annot = True)
```

```python
strong_numeric_cols = ['total day minutes', 'total day charge', 'total eve minutes','total eve charge',
                       'customer service calls','number vmail messages', 'total intl calls']

CT_strong = ColumnTransformer(transformers = [
    ('subpipe_num', subpipe_numeric,strong_numeric_cols),
    ('subpipe_cat', subpipe_cat, cat_cols)
])

logreg_corr = Pipeline([
    ('ct',CT_strong),
    ('logreg', LogisticRegression())
])

logreg_corr.fit(X_train, y_train)

cv_results = cross_validate(
                    estimator=logreg_corr,
                    X=X_train,
                    y=y_train,
                    cv=5,
                    return_train_score=True
)

cv_results['train_score']

plot_roc_curve(logreg_corr, X_test, y_test)

plot_confusion_matrix(logreg_corr, X_test, y_test)

print(classification_report(y_test, logreg_corr.predict(X_test)))
```
Seems like it had the opposite effect, I will try doing a polynomial features on the columns to see if that increases the recall at all.

## Polynomial Feature Model

```python
subpipe_numeric_pf = Pipeline(steps =[
    ('num_impute', SimpleImputer()),
    ('ss', StandardScaler()),
    ('pf', PolynomialFeatures())
])


CT_pf = ColumnTransformer(transformers = [
    ('subpipe_num', subpipe_numeric_pf,numeric_cols),
    ('subpipe_cat', subpipe_cat, cat_cols)
])

logreg_pf = Pipeline(steps = [
    ('ct',CT_pf),
    ('logreg', LogisticRegression())
])

logreg_pf.fit(X_train,y_train)

plot_roc_curve(logreg_pf, X_train, y_train)

plot_confusion_matrix(logreg_pf, X_test, y_test)

plot_roc_curve(logreg_pf, X_test, y_test)

print(classification_report(y_test, logreg_pf.predict(X_test)))
```
Wow, these polynomial features have more than doubled the recall and also increased the accuracy, I'll try increasing the degrees one more time

## Polynomial Feature Degree 3 Model
```python
subpipe_numeric_pf2 = Pipeline(steps =[
    ('num_impute', SimpleImputer()),
    ('ss', StandardScaler()),
    ('pf', PolynomialFeatures(degree = 3))
])


CT_pf2 = ColumnTransformer(transformers = [
    ('subpipe_num', subpipe_numeric_pf2,numeric_cols),
    ('subpipe_cat', subpipe_cat, cat_cols)
])

logreg_pf2 = Pipeline([
    ('ct',CT_pf2),
    ('logreg', LogisticRegression())
])

logreg_pf2.fit(X_train, y_train)

plot_roc_curve(logreg_pf2, X_train, y_train)

plot_confusion_matrix(logreg_pf2, X_test, y_test)

plot_roc_curve(logreg_pf2, X_test, y_test)

print(classification_report(y_test, logreg_pf2.predict(X_test)))
```
A degree of 3 seems to be a good point as the accuracy is starting to decrease and recall didn't increase as much between these two values, for fun let's go one step deeper tho

## Polynomial Features Degree 4 Model
```python
subpipe_numeric_pf3 = Pipeline(steps =[
    ('num_impute', SimpleImputer()),
    ('ss', StandardScaler()),
    ('pf', PolynomialFeatures(degree = 4))
])


CT_pf3 = ColumnTransformer(transformers = [
    ('subpipe_num', subpipe_numeric_pf3,numeric_cols),
    ('subpipe_cat', subpipe_cat, cat_cols)
])

logreg_pf3 = Pipeline([
    ('ct',CT_pf3),
    ('logreg', LogisticRegression(C = .0001))
])

logreg_pf3.fit(X_train, y_train)

plot_roc_curve(logreg_pf3, X_train, y_train)

plot_roc_curve(logreg_pf3, X_test, y_test)

print(classification_report(y_test, logreg_pf3.predict(X_test)))
```
Seems like we should stick with a maximum degree of 3

# Grid Search on Polynomial Features Degree of 3 Model
```python
grid = {
    'logreg__penalty' : ['none','l1','l2','elasticnet'],
    'logreg__max_iter' : [10,100,1000],
    'logreg__C' : [0.000001, 0.00001, 0.0001, 0.001],
    'logreg__tol' : [.0001, .001, .01, .1],
    'logreg__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

gs = GridSearchCV(logreg_pf2, param_grid = grid, verbose =1, cv = 3)

gs.fit(X_train,y_train)

print(gs.best_params_)
print(gs.best_score_)
```
## Best Model with grid search parameters #1
```python
CT_cols = ColumnTransformer(transformers=[
    ('num_cols',subpipe_numeric_pf2, numeric_cols),
    ('ohe', OneHotEncoder(), cat_cols)
])

best_model = Pipeline([
    ('ct', CT_cols),
    ('logreg', LogisticRegression(C = 1e-05, max_iter= 100, penalty= 'none', solver= 'sag', tol= 0.0001))
])

best_model.fit(X_train, y_train)

plot_roc_curve(best_model, X_test, y_test)

print(classification_report(y_test, best_model.predict(X_test)))
```
## Best Model with grid search parameters #2
```python
logreg_pf2_gs = Pipeline([
    ('ct',CT_pf2),
    ('logreg', LogisticRegression(C = 1e-05, max_iter= 100, penalty= 'none', solver= 'sag', tol= 0.0001))
])

logreg_pf2_gs.fit(X_train,y_train)

plot_roc_curve(logreg_pf2_gs, X_test, y_test)

print(classification_report(y_test, logreg_pf2_gs.predict(X_test)))

plot_confusion_matrix(logreg_pf2_gs, X_test, y_test)
```
# Interpreting coefficients
```python
categorical_feature_names = CT_pf.named_transformers_['subpipe_cat'].named_steps['ohe'].get_feature_names(input_features = cat_cols)
poly_feature_names = CT_pf.named_transformers_['subpipe_num'].named_steps['pf'].get_feature_names(input_features = numeric_cols)
allcols = poly_feature_names + list(categorical_feature_names)

# Getting the feature names to relate to the coefficients

# Making a  model to pull out the coefficients for the model

coefs = LogisticRegression(C = 1e-05, max_iter= 100, penalty= 'none', solver= 'sag', tol= 0.0001)

# Coefficients labeled and converted to % change on churn

pflist = list(zip(allcols, (np.exp(logreg_pf.named_steps['logreg'].coef_[0]) - 1)  *100))
sorted(pflist,key = lambda x:x[1])
```
Seems like the international plan and voicemail plan have the largest change between being, the states seem to have a big impact but I don't know how to interpret that with the data I have currently.  The amount of customer service calls also seems to add quite a bit of a chance for subscribers to churn.
