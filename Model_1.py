import numpy as np 
import pandas as pd 

# sklearn : machine leanring and deepleanring 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error


# df = pd.read_csv(r'C:\Users\DELL\Desktop\Batches\Data Science - 11am\Data sets\Student_Marks.csv')

df = {
    'number_courses': [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,7,8,8],
    'time_study':     [1,2,3,1,2,3,2,3,4,2,3,5,3,4,5,4,5,6,6,7],
    'Marks':[45,55,60,50,60,68,62,70,75,65,72,80,70,78,85,82,88,92,95,98]
}

df = pd.DataFrame(df)
print(df.head())
print(df.info())
print(df)



# feactures selections 
# x1,x2 , y

X = df[['number_courses','time_study']].dropna()
y = df[['Marks']].dropna()


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


objLinear = LinearRegression()

objLinear.fit(x_train,y_train)


prediction = objLinear.predict(x_test)

print(r2_score(y_test,prediction))


myprediction = objLinear.predict([[2,2]])
print(myprediction)