#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MinMaxScaler import minmax_scaler
from hypothesis import hyp
from gradient import grad
from Performance_analysis import *
from math import sqrt
from sklearn.model_selection import train_test_split

#load dataset
df = pd.read_csv(r'C:\Users\Xero\Desktop\Ai\Dataset\Profit.csv')
df = df.replace({"New York": 1, "California": 2, "Florida": 3})
print(df.head())
df1 = df.copy()

#rescale data
for feat in df1.columns:
    df1[feat] = round(minmax_scaler(df1,feat,f_range = (1,2)),3)

print(df1.head())

#input and target
X = df1.drop(columns = ['Profit']).to_numpy()
X = np.matrix(X)
y = df1['Profit'].to_numpy()
y = np.matrix(y).T

#size
m,n = X.shape   #m = sample size, n = no of features
_,k = y.shape   #k = no of classes

X0 = np.ones((m,1))              #create coeficient of intercept
X = np.c_[X0,X]            #add 1s' column to 1st column of X
init_theta = np.zeros((n+1,k))   #initialize theta as zeros

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1, 
                                                 random_state = 0)

#size
m,n = X_train.shape   #m = sample size, n = no of features
_,k = y_train.shape   #k = no of classes

#train model
J_hist ,theta = grad(X_train, y_train, init_theta,alpha=0.2,
                     lamb=0,max_iter=5000)
print()
print("Trained Theta")
print("----------------------")
print(theta)
print()
print(J_hist[-1])

plt.plot(np.arange(len(J_hist)),J_hist, lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')

#predict y values
y_pred = hyp(X_test,theta)

#rescale predicted y values to calculate mean square error
re_range = (df['Profit'].min(),df['Profit'].max())
y_list = list()
for i in y_pred.tolist():
    y_list.extend(i)
ydf = pd.DataFrame({"Profit": y_list})
ydf = minmax_scaler(ydf,'Profit',f_range=re_range)

#rescale actual y values
y_test_re = list()
for i in y_test.tolist():
    y_test_re.extend(i)
y_test_df = pd.DataFrame({"Profit": y_test_re})
y_test_df = minmax_scaler(y_test_df,'Profit',f_range=re_range)

print("Predicted Y Value")
print("----------------------------------")
print(ydf.head())
print('----------------------------')
print("Actual Y-value")
print(y_test_df.head())
print('----------------------------')
print()

mserr = mse(y_test_df.to_numpy(),ydf['Profit'].to_numpy())
print("mean error: ",sqrt(mserr))

r2_score = r_2(ydf.to_numpy(),y_test_df.to_numpy())
print("R2 Score: ", r2_score)

'''
test_theta = np.matrix([[-0.02208209584605561],
                        [0.8994177],
                        [0.02157384],
                        [0.09543082],
                        [0.00328713]])

y_pred = hyp(X_test, test_theta)
y_list = list()
for i in y_pred.tolist():
    y_list.extend(i)
ydf = pd.DataFrame({"Profit": y_list})
ydf = minmax_scaler(ydf,'Profit',f_range=re_range)

mserr = mse(y_test_df.to_numpy(),ydf['Profit'].to_numpy())
print("mean error: ",sqrt(mserr))

r2_score = r_2(ydf.to_numpy(),y_test_df.to_numpy())
print("R2 Score: ", r2_score)
'''