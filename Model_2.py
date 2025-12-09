#ridge regression 
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# load data set in a our file 
loadedData = pd.read_csv(r'C:\Users\DELL\Desktop\Batches\Data Science - 11am\Data sets/bus_100.csv')
dataFrame = pd.DataFrame(loadedData)
# print(dataFrame.head())
# print(dataFrame.describe())
# print(dataFrame.info())
# convert text data into numeric data 
Encoder = LabelEncoder()
dataFrame['Agency'] = Encoder.fit_transform(dataFrame['Agency'])
dataFrame['Source'] = Encoder.fit_transform(dataFrame['Source'])
dataFrame['Destination'] = Encoder.fit_transform(dataFrame['Destination'])
dataFrame['Bus Type'] = Encoder.fit_transform(dataFrame['Bus Type'])
x = dataFrame[['Agency','Source','Destination','Bus Type','Duration (hours)','Total Seats']].dropna()
y = dataFrame[['Fare Price (INR)']].dropna()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# create object of ridge regression 
Bridge = Ridge()
Bridge.fit(x_train,y_train)
y_predict = Bridge.predict(x_test)
r2score = r2_score(y_test,y_predict)
mse = mean_squared_error(y_test,y_predict)
print(r2score)
print(mse)