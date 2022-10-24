# Originally created: 24th October, 2022
#                     Deepawali
# Last amended: 09th Oct, 2021
# Myfolder: C:\Users\123\OneDrive\Documents\streamlit\churnapp
#	         /home/ashok/Documents/churnapp
#           VM: lubuntu_healthcare
# Ref: https://builtin.com/machine-learning/streamlit-tutorial
#
# Objective:
#             Deploy an ML model on web
#
########################
# Notes:
#       1, Run this app in its folder, as:
#          cd /home/ashok/Documents/churnapp
#          streamlit  run  churn-app.py
#       2. Accomanying file to experiment is
#          expt.py
########################


# 1.0 Call libraries
# Install as: pip install streamlit
# Better create a separate conda environment for it
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn


#1.1 When manually trying to discover bugs,
#     set debugging to True. But when running
#      the app, set debugging to False:

debugging = True  # Else ,False


# 1.2
# We need version numbers in 'requirements.txt' file:
    
if debugging:
    print(st.__version__)
    print(np.__version__)
    print(pd.__version__)
    print(sklearn.__version__)
    
    




# 2.0 Write some body-text on the Web-Page:

st.write("""
# Churn Prediction App

Customer churn is defined as the loss of customers after a certain period of time.
Companies are interested in targeting customers who are likely to churn. They can
target these customers with special deals and promotions to influence them to stay
with the company.

This app predicts the probability of a customer churning using Telco Customer data. Here
customer churn means the customer does not make another purchase after a period of time.
""")





# 3.0 Create a function to display
#      widgets to receive user inputs:
#       This will be our test data:
    
def user_input_features():
    # 3.1 Create four widgets
    gender         = st.sidebar.selectbox('gender',('Male','Female'))
    PaymentMethod  = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0,118.0, 18.0)
    tenure         = st.sidebar.slider('tenure', 0.0,72.0, 0.0)
    # 3.2 Collect widget output in a dict format
    data = {
            'gender':        [gender],         # Should be a list data structure
            'PaymentMethod': [PaymentMethod],
            'MonthlyCharges':[MonthlyCharges],
            'tenure':        [tenure]
            }
    # 3.3 Transform dict to DataFrame
    data1 = pd.DataFrame(data)
    # 3.4 Return test dataframe
    return data1


if debugging:
    import os
    os.chdir("C:\\Users\\123\\OneDrive\\Documents\\streamlit\\churnapp")


# 4.0 Read train data from current folder
#     Default folder is where streamlit
#     is being run. So this file
#     should be in /home/ashok/Documents/churnapp
#     Else, home folder is the default.

train = pd.read_csv("telco_churn.csv")


if debugging:
    print(train.head())


# 4.1 We will select only a few columns
#     for our model:
    
cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'tenure', 'PaymentMethod', 'MonthlyCharges', 'Churn']
train = train[cols]


if debugging:
    print(train.head())




# 5.0 Widget to download file:
# Ref: https://docs.streamlit.io/library/api-reference/widgets/st.download_button    
# IMPORTANT: Cache the conversion to prevent computation on every rerun
# Ref: https://docs.streamlit.io/library/advanced-features/caching
# Note that the data to be downloaded is stored in-memory while the user
#   is connected, so it's a good idea to keep file sizes under a couple 
#     hundred megabytes to conserve memory.


# 5.1 Make this file also available for download:
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# 5.2
csvstr = convert_df(train)

# 5.3
st.download_button(
                   label="Download train data as CSV",  
                   data=csvstr,
                   )



# 6.0 Create a component to upload test file in the sidebar.
uploaded_file = st.sidebar.file_uploader(
                                          "Upload your input CSV file",
                                           type=["csv"]
                                         )


# 6.1 Read data from test file. Else, read from widgets
if uploaded_file is not None:
    # 6.2 Read the 'uploaded_file' as data frame:
    test_df = pd.read_csv(uploaded_file)
else:
    # 6.3 Call the function and get a 1-row DataFrame
    test_df = user_input_features()



if debugging:
    test_df.head()



# 7.1 First fill up NAs in it:
train.fillna(0, inplace=True)
train = train.drop(columns=['Churn'])



# 7.2 Stack vertically 1-row test data 
#     with train data. We need this stacking
#     becuase we do not have any model to transform
#     cat to OHE. Just with one row, pd.get_dummies()
#     will not work.
    
test_train = pd.concat(
                        [
                          test_df,
                          train
                        ],
                        axis=0
                      )




if debugging:
    test_train.shape
    

# 7.3 
#Transform test_train to dummy features
# 7.3.1 Our cat columns


cat_cols = ['gender','PaymentMethod']
test_train = pd.get_dummies(
                            test_train,
                            prefix = cat_cols,
                            columns = cat_cols
                            )



if debugging:
    test_train.columns
    

# 7.4 Just read the first row:
    
test_train = test_train[:1]      # Selects only the first row (the user input data)
test_train.fillna(0, inplace=True) # Fill all NaNs with zeros


if debugging:
    test_train.columns

    
# 7.5 What are our OHE feature names?
#       Eight features, in all:

features = ['MonthlyCharges',
            'tenure',
            'gender_Female',
            'gender_Male',
            'PaymentMethod_Bank transfer (automatic)',
            'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check'
            ]


# 8.0 Get a data subset of our features:
test_train = test_train[features]


# 8.1 Displays the user input features
st.subheader('User test features')
#print(test_train.columns)


# 8.2
if uploaded_file is not None:
    # 6.2.1 Write the first row
    st.write(test_train)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(test_train)



# 9.0 Read in saved classification model
#     from the current folder:
load_clf = pickle.load(open('churn_clf.pkl', 'rb'))



# 9.1 Apply model to make predictions
prediction = load_clf.predict(test_train)               # 'prediction' will have value of 1 or 0
prediction_proba = load_clf.predict_proba(test_train)   # Prediction probability



# 10.0 Display Labels
st.subheader('Prediction')
churn_labels = np.array(['No','Yes'])  # churn_labels is an array of strings
                                       # churn_labels[0] is 'No' and churn_labels[1] is 'Yes'
st.write(churn_labels[prediction])     # Display 'Yes' or 'No' depending upon value of
                                       # 'prediction'


# 10.1 Also display probabilities
st.subheader('Prediction Probability')


# 8.2 Numpy arrays are displayed with column names
#     as 1 or 0
st.write(prediction_proba)
######################################
