#import numpy(optional) as np and pandas as pd
import numpy as np
import pandas as pd
#read the excel data as a pandas dataframe
stroke = pd.read_excel('H:\Machine learning\strokes.xlsx')



#convert the marriage data to boolean values
stroke2 = pd.get_dummies(stroke['ever_married'])
stroke = pd.concat([stroke, stroke2], axis=1).reindex(stroke.index)
stroke.rename(columns = {'No':'married_no',
                              'Yes':'married_yes'}, inplace = True)

#convert the profession data to boolean values
stroke3 = pd.get_dummies(stroke['work_type'])
stroke = pd.concat([stroke, stroke3], axis=1).reindex(stroke.index)


#convert the residence data to boolean values
stroke4 = pd.get_dummies(stroke['Residence_type'])
stroke = pd.concat([stroke, stroke4], axis=1).reindex(stroke.index)


#convert the gender data to boolean values
stroke5 = pd.get_dummies(stroke['gender'])
stroke = pd.concat([stroke, stroke5], axis=1).reindex(stroke.index)


#convert the smoking data to boolean values
stroke6 = pd.get_dummies(stroke['smoking_status'])
stroke = pd.concat([stroke, stroke6], axis=1).reindex(stroke.index)



#set the data to be trained on as X and the data to be predicted  as y. In this case, we're going to predict if the patient has a stroke or not. So I'll set the 'stroke' column as y.
y = stroke['stroke']
X = stroke[['age', 'hypertension', 'heart_disease','avg_glucose_level', 'bmi','married_no', 'married_yes', 'Govt_job',
       'Never_worked', 'Private', 'Self-employed', 'children', 'Rural',
       'Urban', 'Female', 'Male', 'Unknown', 'formerly smoked', 'never smoked',
       'smokes']]

#import train_test_split to split the data into training and testing groups
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


#import LogisticRegression and fit the training data
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train, y_train)


#make predictions on the testing data
predictions = lm.predict(X_test)


#check the accuracy(precision of classification report should be around 0.70)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))