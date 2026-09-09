import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
# X shape: (m, N) -> m rows, N features
# W shape: (N, 1) -> N weights
# b shape: scalar float (or bias)
# comments are treated as notes for me to revieew later


def scale(data):
    #scale the given data
    x_mean=np.mean(data, axis=0)
    x_std=np.std(data, axis=0)
    x_std[x_std == 0] = 1e-8
    x_scaled=(data-x_mean)/x_std
    return x_scaled,x_mean,x_std
def rmse(obs,pred):# root mean square obs for observed and pred for predicted
    error=0.0
    for y,y_hat in zip(obs,pred):
        error=error+(y-y_hat)**2
    merror=error/float(len(obs))
    return merror**0.5
# now to find slope and intercept function essentially plotting the line y=mx+c
def predict(x,w,b):#predection function
    return np.dot(x,w) +b
def costfx(x,y,w,b):
    m=len(x) # since we have to divide by no of out puts
    predections=predict(x,w,b)
    cost=(1/(2*m))*np.sum((predections-y)**2)
    return cost
#check obsidian for why we added 2 in the denominator essentially we can didvide by 2 here 
#or it will cancel out during dot product in dw
def graddes(x,y,lr,epochs):
    m,features=x.shape # to give rows and columns 
    w=np.zeros((features,1))# to make w as a compatible matrix to multiply 
    b=0.0# standard becuase it will change with grad descent 
    cost_history=[]
    for i in range(epochs): # epochs are training iterations
        y_pred=predict(x,w,b)
        error=y_pred-y
        dcost_dw=(1/m)*np.dot(x.T,error)
        dcost_db=(1/m)*np.sum(error)
        w-=lr*dcost_dw
        b-=lr*dcost_db
        if i%50==0:
            print(i)
        cost_history.append(costfx(x,y,w,b))
    return w,b,cost_history


#model training #
df=pd.read_csv('weatherHistory.csv')
print("Columns: ", df.columns.tolist())#lists every column name in your CSV as a standard Python string list.
#check obsidian for further details
feature_cols = [
    'Apparent Temperature (C)', 
    'Humidity', 
    'Wind Speed (km/h)', 
    'Wind Bearing (degrees)', 
    'Visibility (km)', 
    'Pressure (millibars)'
]

target_col = 'Temperature (C)'# we wish to rpedict temprature
#essentially we are giving each columns its own vaaraible column matrix = each feature and variable
# Extract X and y as NumPy arrays

X_raw = df[feature_cols].values#Filters the Pandas DataFrame to keep only the 6 chosen feature columns in an array columns.
y = df[[target_col]].values  # Double brackets [[ ]] keep shape as (m, 1) and turning it into a numpy matrix

#single bracket returns an array of each feature coln
#df[[target_col]]: The inner double brackets [[ ]] select the target column while telling Pandas to keep it as a 2D DataFrame rather than flattening it into a 1D Series.
#.values: Converts the Pandas DataFrame into a raw NumPy array.



xscale,xmean,xstd=scale(X_raw)
lr=0.001
epochs=5000
w,b,cost_history=graddes(xscale,y,lr,epochs) 

ypred=predict(xscale,w,b)
rmse_val=rmse(y,ypred)


#unsclaing the orignal data 
apparent_temp_raw = X_raw[:, 0]

plot.figure(figsize=(8, 5))

# 1. Scatter actual data points
plot.scatter(apparent_temp_raw, y, color='blue', alpha=0.1, label='Actual Data')

# 2. Scatter predicted data points
plot.scatter(apparent_temp_raw, ypred, color='red', alpha=0.1, label='Model Predictions')

plot.xlabel('Apparent Temperature (°C)')
plot.ylabel('Actual Temperature (°C)')
plot.title('Apparent Temperature vs Temperature')
plot.legend()
plot.grid(True)
plot.show()

