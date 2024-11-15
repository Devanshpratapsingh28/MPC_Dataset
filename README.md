# Mobile_Price_Classification_Dataset

## Problem Statement :
Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.

He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.

Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.

In this problem you do not have to predict actual price but a price range indicating how high the price is.

## Dataset Source : https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification

### Libraries Used 
1. Numpy
2. Pandas
3. Matplotlib
4. Seaborn
5. Scikit Learn (Sklearn)

### Observations and Data Preprocessing Steps
1. Dropped `id` columnn as it no use here and also it is null.
2. `pc and fc` and `3g and 4g` have slightly more correlation with each other in comarison to others.
3. Total four categories of price range is there.
4. Converting datypes of some variables like `blue`,`four_g`, `n_cores` etc. to category so that later we can easily create pipelines with different properties for numerical and categorical columns.
5. performed a train-test split using **train_test_split** from **sklearn.model_selection**.
6. Scale the inputs using **Standard Scalar**. 

### Models Tried 
1. Logistic Classification and its varients like ridge,lasso
2. Stochastic Gradient Descent(SGD) Classifier
3. DecisionTreeClassifier
4. RandomForestClassifier
5. Support Vector Machine Classifier(SVC)

### Accuracy Table

Below table is showing accuracy score of various models used in this task:

| Model                   | Accuracy Score     |  
|-------------------------|--------------------|
| Ridge Classifier        | 0.632              |
| SGD                     | 0.820              |
| Decision Tree           | 0.838              |
| SVM Classifier          | 0.895              |
| Random Forest           | 0.900              |
| **Logistic Regression** | **0.985**          |
| **Lasso Classifier**    | **0.985**          |

### Best Model : Logistic Regression and Lasso Regression
### Note : 
  - I prefer `Lasso Regression` here because it performs `Feature Selection`.
  - I have tried to fine tune using `GridSearchCV` but get less than `98.5 %` accuracy.
