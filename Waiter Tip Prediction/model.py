import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

data = pd.read_csv(r"G:\VS Code\projects\ML-Projects\Waiter Tip Prediction\tip_data.csv")
x = data.drop(columns = ['tip'])
y = data.iloc[:,1]

ohe = OneHotEncoder(dtype=np.int32)
sex_time = ohe.fit_transform(x[['sex', 'time']]).toarray()
sex_time = pd.DataFrame(sex_time, columns = ['female', 'male', 'dinner', 'lunch'])

oe = OrdinalEncoder(categories = [['No', 'Yes'], ['Sun', 'Thur', 'Fri', 'Sat']])
smoker_day = oe.fit_transform(x[['smoker', 'day']])
smoker_day = pd.DataFrame(smoker_day, columns=['smoker', 'Day'])

new_x = pd.concat([x, sex_time, smoker_day], axis = 1)

new_x.drop(columns = ['sex', 'smoker', 'day', 'time', 'female', 'lunch'], inplace = True)

x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)

pickle.dump(lr, open('model.pkl', 'wb'))
