# What is Lasso Regression ?
# Lasso Regression is a regression method based on the absolute shrinkage  and selection operators

# Ridge regression |
# 1. When all features are useful 
# 2. prevent overfitting but doesnot remove feactures  


# Lasso regression 
# 1.  when you suspect some features are not important 
# 2.  Automatically removes irrelevent feactures

# # House Prediction 
# house  | area | rooms | price   | Garden size 
# 1      | 1000 |   3   | 50 lakh | 12
# 2      | 900  |   2   | 45 lakh | 10
# 3      | 1100 |   2   | 55 lakh | 13 
# 4      | 1200 |   3   | 65 lakh | 11
# 5      | 1300 |   4   | 75 lakh | 12
        # 0.045    1.4            0.00008     
# Area -> rooms ---> price 
# Add Garden size ----> garden
# Ridge regression (All feac....) -> overfitting 
# Lasso regression 

import numpy as np
from sklearn.linear_model import Ridge,Lasso
# Array of Data 
x = np.array([
    [1000,3,12],
    [900,2,10],
    [1100,3,13],
    [1200,4,11],
    [800,2,9],
])
y = np.array([50,45,55,65,40]) 
# x = [12.00098,15.9087, 16.9087] ,,, 0 
# 0.5 => [10.5,14.3,15.2]
#  1 => [9.8, 13.9, 14.6 ]
RidgeObj  = Ridge(alpha=1)
RidgeObj.fit(x,y)
print("Ridge Coeff : ",RidgeObj.coef_)
#  y = Bx1+ Bx2 + Bx3
# Lasso Model 
lasso= Lasso(alpha=1)
lasso.fit(x,y)
print("Lasso Coeff : ",lasso.coef_)
