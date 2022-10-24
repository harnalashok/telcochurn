# Last amended: 23rd October, 2022
# My folder: /home/ashok/Documents/churnapp
#	         D:\data\OneDrive\Documents\streamlit\churnapp
# VM: lubuntu_healthcare
# Objectives:
#           i)   Develop a churn model
#           ii)  Use it in a webapp
#           iii) How to use classifier: 'HistGradientBoostingClassifier()'

# Ref: Github: https://github.com/spierre91/medium_code/tree/master/streamlit_builtin

# 1.0 Call libraries
#     No streamlit library:
    
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle
import os


# 1.1 Some options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# 2.0 Read data 
#path = "/home/ashok/Documents/churnapp/"
path = "C:\\Users\\123\\OneDrive\\Documents\\streamlit\\churnapp\\"

# 2.1
data = pd.read_csv(path+'telco_churn.csv')
data.shape    # (7043,21)
data.columns

# 2.2 Any Nulls?
data.isnull().sum()   

# 2.3 Class distribution
data['Churn'].value_counts(normalize = True)

# 2.4 Data types. Most are objects
data.dtypes



# 3.0 Select just few columns for our model
df = data[['gender', 'PaymentMethod', 'MonthlyCharges', 'tenure', 'Churn']].copy()

# 3.1
df.head()


"""
Convert categorical variable into dummy/indicator variables.
pandas.get_dummies(
                   data,
                   prefix=None,
                   prefix_sep='_',
                   dummy_na=False,
                   columns=None,
                   sparse=False,
                   drop_first=False,
                   dtype=None
                   )


"""
# 3.2 Example of One Hot Encoding:
dx = pd.DataFrame([
                   ['a','r',23],
                   ['a','s',24],
                   ['b','s',50],
                   ['b','s',78],
                   ['b','s',90]
                  ],
                 columns = ['x', 'y', 'z']
                 )
dx
# 3.3 Note columns 'x' and 'y' are dropped, 
#      but not col 'z':
    
pd.get_dummies(dx, prefix = "col") 


# 3.4 Convert cat variables into dummy variables:
cat_cols = ['gender','PaymentMethod']
df = pd.get_dummies(df, prefix = cat_cols , columns = cat_cols)


# 3.5 Check
df.head()


# 4.0 Transform, target, Churn to 1 or 0
#     We can use df.map() also:
#       df['Churn'].apply(target_encode)    

df['Churn'] = np.where(
                       df['Churn']=='Yes',
                       1, 0)   


# 5.0 Separate into predictor and target
X = df.drop('Churn', axis=1)
Y = df['Churn']


# 5.1 Our features are:
features  = list(X.columns)
features

len(features)     # 8

# 5.2 Build simple model:
    
clf = HistGradientBoostingClassifier(
                                      learning_rate=0.1,
                                      max_iter=100,
                                      max_depth = 80
                                      
                                     )

# 5.3 Train the model:
clf.fit(X, Y)

# 5.4 Store model to disk:
pickle.dump(clf, open(path+'churn_clf.pkl', 'wb'))

# 6.0 Load model and predict (just to check):

clx = pd.read_pickle(path+'churn_clf.pkl') 

# 6.1 Make predictions:   
y_pred = clx.predict(X)    
np.sum( Y == y_pred)/len(Y)  # 83% for train data
################# DONE ###############