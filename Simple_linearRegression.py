from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load csv file
df = pd.read_csv('C:\\Users\\toshiba\\Downloads\\archive (2)\\simple house price prediction.csv')
print(df)

#show data on chart
plt.scatter(df[['area']],df.price)
plt.show()

#Model training
sml = linear_model.LinearRegression()
sml.fit(df[['area']],df.price)
# prediction of the whole column
pred = sml.predict(df[['area']])
# And then add it to the existing file to compare the actual values and predicted values
df['Pred_col']=pred
print(df)
# Check score of the data
print(sml.score(df[['area']],df.price))
#give value for prediction
print(sml.predict([[5000]]))
print(sml.coef_)
print(sml.intercept_)
check = 253.5*5000 -32261.11111111089
print(check)

# Visualization
# plt.xlabel('Area',color='b',fontsize=12)
# plt.ylabel('Price',color='b',fontsize=12)
# plt.title('House Price Prediction',color='g',fontsize=14,fontweight='bold')
# plt.scatter(df[['area']],df.price,color='tab:blue',label='Actual Values')
# plt.plot(df.area,sml.predict(df[['area']]),color='r',label='Predicted line')
# plt.legend(frameon=True,facecolor='white',edgecolor='black')
# plt.grid(True,linestyle='--',alpha=0.5,color='lightgrey')
# plt.show()