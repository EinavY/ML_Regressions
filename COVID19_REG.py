
import datetime
import pandas as pd
import numpy as np
import json
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
from numpy import arange
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


#data
data = r'C:\Einav\COVID-19_Daily_Testing_-_By_Test.csv'
df = pd.read_csv(data, encoding = "ISO-8859-1")
df = df.sort_values(by="Date")
dataF = pd.DataFrame(df)
x = df.iloc[:, 5:23].values
y_positive = df.iloc[:, 2:3].values
y_not_positive = df.iloc[:, 3:4].values
x_df = pd.DataFrame(x)


#vizualization
for i in x_df:
    b= x_df.iloc[:, i]
    plt.scatter(b,y_positive, color='purple')
    plt.scatter(b, y_not_positive, color = 'pink')
    plt.show()
    i = i+1

#clean the data
df.replace(['nan', 'None'] , np.nan)
df.fillna(value=np.nan, inplace=True)
df.replace(to_replace ='nan', value=np.nan, inplace=True)
df = df.dropna(axis=0,how='all')
df = df.dropna(axis=1,how='all')

#BackWAERD- fitured selection
#Take From- https://www.analyticsvidhya.com/blog/2020/10/a-comprehensive-guide-to-feature-selection-using-wrapper-methods-in-python/
def backward_elimination(data, target,significance_level = 0.05):
    features = data.columns.tolist()
    while(len(features)>0):
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(y_positive, features_with_constant).fit().pvalues[1:]
        max_p_value = p_values.max()
        if(max_p_value >= significance_level):
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features.pd.DataFrame

##Cross validation
def cross_Validation (model,x,y):
    accuracies = cross_val_score(model,x,y,cv=10) 
    data_cv = [accuracies.mean()]
    ##data_cv.append([accuracies.std()]) 
    return data_cv

##split data to train and test
x_train, x_test, y_train_positive, y_test_positive = train_test_split(x_df, y_positive, test_size = 0.2, random_state = 0)
x_train, x_test, y_train_not_positive, y_test_not_positive = train_test_split(x_df, y_not_positive, test_size = 0.2, random_state = 0)


##standartization
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
##sns.kdeplot(x_df[:,2], fill=True, color = 'violet' )


##evalution- function that appriciate the model
## take from https://towardsdatascience.com/linear-regression-in-python-9a1f5f000606
def evaluation(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    data_ev = [mse, rmse, r2]
    return data_ev

summery_df_positive = pd.DataFrame()#df summery
summery_df_not_positive = pd.DataFrame()#df summery

##linear regression
def linear_regression(x_train,x_test,y_train,y_test, summery_df):
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    cv = cross_Validation(regressor,x_train,y_train)
    ev = evaluation(y_test, y_pred)
    summarylinear = pd.DataFrame([ev], columns = ['mse', 'rmse', 'r2'])
    summarylinear['cross validation-mean']= cv
    summery_df = pd.DataFrame()
    summery_df = summery_df.append(summarylinear,ignore_index=True)
    sns.regplot(y_test, y_pred)
    return summery_df

#positive
linearReg = linear_regression(x_train, x_test, y_train_positive, y_test_positive, summery_df_positive)
summery_df_positive = summery_df_positive.append(linearReg, ignore_index=True)

#not positive
linearReg = linear_regression(x_train, x_test, y_train_not_positive, y_test_not_positive, summery_df_not_positive)
summery_df_not_positive = summery_df_not_positive.append(linearReg, ignore_index=True)


##polynomial regression
def polynomial_regression(x_train,x_test,y_train,y_test,summery_df):
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly_train = poly_reg.fit_transform(x_train)
    X_poly_test = poly_reg.transform(x_test)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly_train, y_train)
    y_pred = lin_reg_2.predict(X_poly_test)
    cv = cross_Validation(lin_reg_2, X_poly_train, y_train)
    ev = evaluation(y_test, y_pred)
    summaryPoly = pd.DataFrame([ev], columns = ['mse', 'rmse', 'r2'])
    summaryPoly['cross validation- mean'] = cv
    summery_df = pd.DataFrame()
    summery_df = summery_df.append(summaryPoly, ignore_index=True)
    sns.regplot(y_test, y_pred)
    return summery_df

#positive
polyReg = polynomial_regression(x_train,x_test,y_train_positive,y_test_positive,summery_df_positive)
summery_df_positive = summery_df_positive.append(polyReg, ignore_index=True)

#not positive
polyReg = polynomial_regression(x_train,x_test,y_train_not_positive,y_test_not_positive,summery_df_not_positive)
summery_df_not_positive = summery_df_not_positive.append(polyReg, ignore_index=True)

##ridge regression
def ridg_regression(x_train,x_test,y_train,y_test,summery_df):
    #cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    modelR = Ridge(alpha= 0.05, tol=1e-3, random_state=2)
    modelR.fit(x_train, y_train)
    y_pred = modelR.predict(x_test)
    cv= cross_Validation(modelR, x_train, y_train)
    ev = evaluation(y_test, y_pred)
    summeryRidge = pd.DataFrame([ev], columns = ['mse', 'rmse', 'r2'])
    summeryRidge['cross validation-mean'] =cv
    summery_df = pd.DataFrame()
    summery_df = summery_df.append(summeryRidge, ignore_index=True)
    sns.regplot(y_test, y_pred)
    return summery_df

#positive
ridgReg = ridg_regression(x_train, x_test, y_train_positive, y_test_positive, summery_df_positive)
summery_df_positive = summery_df_positive.append(ridgReg, ignore_index=True)

#not positive
ridgReg = ridg_regression(x_train, x_test, y_train_not_positive, y_test_not_positive, summery_df_not_positive)
summery_df_not_positive = summery_df_not_positive.append(ridgReg, ignore_index=True)


##lasso regression
def lasso_regression(x_train,x_test,y_train,y_test,summery_df):
    #cv2 = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    modell = Lasso(alpha= 0.05, tol=1e-3, random_state=2)
    modell.fit(x_train, y_train)
    y_pred = modell.predict(x_test)
    cv= cross_Validation(modell,x_train, y_train)
    ev = evaluation(y_test, y_pred)
    summeryLasso = pd.DataFrame([ev], columns = ['mse', 'rmse', 'r2'])
    summeryLasso['cross validation-mean'] =cv
    summery_df = pd.DataFrame()
    summery_df = summery_df.append(summeryLasso, ignore_index=True)
    sns.regplot(y_test, y_pred)
    return summery_df

#positive
lassoReg = ridg_regression(x_train, x_test, y_train_positive, y_test_positive, summery_df_positive)
summery_df_positive = summery_df_positive.append(lassoReg, ignore_index=True)

#not positive
lassoReg = ridg_regression(x_train, x_test, y_train_not_positive, y_test_not_positive, summery_df_not_positive)
summery_df_not_positive = summery_df_not_positive.append(lassoReg, ignore_index=True)

##random forest regressiom
def random_forest_regression(x_train, x_test, y_train, y_test, summary_df):
    random_forest=RandomForestRegressor(n_estimators=100, random_state=2)
    random_forest.fit(x_train, y_train)
    y_pred = random_forest.predict(x_test)
    cv= cross_Validation(random_forest,x_train, y_train)
    ev = evaluation(y_test, y_pred)
    summeryRandom = pd.DataFrame([ev], columns = ['mse', 'rmse', 'r2'])
    summeryRandom['cross validation-mean'] =cv
    summery_df = pd.DataFrame()
    summery_df = summery_df.append(summeryRandom, ignore_index=True)
    sns.regplot(y_test, y_pred)
    return summery_df

#positive
randomReg = random_forest_regression(x_train, x_test, y_train_positive, y_test_positive, summery_df_positive)
summery_df_positive = summery_df_positive.append(randomReg, ignore_index=True)

#not positive
randomReg = random_forest_regression(x_train, x_test, y_train_not_positive, y_test_not_positive, summery_df_not_positive)
summery_df_not_positive = summery_df_not_positive.append(randomReg, ignore_index=True)

##knn regression
def knn_regression(x_train, x_test, y_train, y_test, summery_df):
    classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p = 2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    cv= cross_Validation(classifier,x_train, y_train)
    ev = evaluation(y_test, y_pred)
    summeryKnn = pd.DataFrame([ev], columns = ['mse', 'rmse', 'r2'])
    summeryKnn['cross validation-mean'] =cv
    summery_df = pd.DataFrame()
    summery_df = summery_df.append(summeryKnn, ignore_index=True)
    sns.regplot(y_test, y_pred)
    return summery_df

#positive
knnReg = knn_regression(x_train, x_test, y_train_positive, y_test_positive, summery_df_positive)
summery_df_positive = summery_df_positive.append(knnReg, ignore_index=True)

#not positive
knnReg = knn_regression(x_train, x_test, y_train_not_positive, y_test_not_positive, summery_df_not_positive)
summery_df_not_positive = summery_df_not_positive.append(knnReg, ignore_index=True)

    



    
    
    
    
   
