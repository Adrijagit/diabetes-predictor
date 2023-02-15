#creating svm model 
import numpy as np
class svm_classifier():

    #initialising the hyperparameters
    def __init__(self,learning_rate,no_of_iterations,lambda_parameter):
        self.learning_rate=learning_rate
        self.no_of_iterations=no_of_iterations
        self.lambda_parameter=lambda_parameter
        
    #fitting the dataset to svm classifier   
    def fit(self,X,Y):
        
        #m defines the no. of rows-->no. of data points
        #n defines the no. of columns-->no. of input features

        self.m,self.n=X.shape
     #w will store the weight values and as it is basically an array of size equal to n,so we use self.n and initializing them by zeros
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.Y=Y
    #implementing gradient descent algo for optimisation
        for i in range(self.no_of_iterations):
            self.update_weight()

    #function for updating the weight and bias values
    def update_weight(self):
        #label encoding,acts more like a ternery operator
        #as in our dataset we have 0 and 1 ,as label but for svm we need 1 and -1,so converting 0 to -1
        y_label=np.where(self.Y<=0,-1,1)
        
        #gradient descent algo
        for index,x_i in enumerate(self.X):
            condition=y_label[index]*(np.dot(x_i,self.w)-self.b)>=1
            #calculating the gradients(dw,db) according to gd algo
            if(condition==True):
                dw=2*self.lambda_parameter*self.w
                db=0
            else:
                dw=2*self.lambda_parameter*self.w-np.dot(x_i,y_label[index])
                db=y_label[index]

            #changing the weight and bias value accordingly
            self.w = self.w-self.learning_rate*dw
            self.b = self.b-self.learning_rate*db


    #predict the label for a given input
    def predict(self,X):

        #euation of the hyperplane -> y=xw-b
        output=np.dot(X,self.w)-self.b

        #now as the value of output can be any positive or negative number,so we need to generalise them in any two categories
        #np.sign assign -1 or +1 to labels ,soley onif number is negative or positive irrespective of its original value
        #as in svm we finally need to catergorise data in two (in our dataset 1 or 0)

        predicted_label=np.sign(output)
        y_hat=np.where(predicted_label<=-1,0,1)
        return y_hat



#creating an instance of the class
model =svm_classifier(learning_rate=0.001,no_of_iterations=1000,lambda_parameter=0.01)

#importing other dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#data collection and preprocessing

#loading the data from the csv file to the pandas dataframe
diabetes_data=pd.read_csv("C:/Users/KALYAN KUMAR GUHA/Desktop/college stuff/PROJECTS/pima diabetes.csv")

#printing the first five rows of the dataframe
print("\n First five rows of the dataset")
print(diabetes_data.head())
print("\n Total number of rows and columns")
print(diabetes_data.shape)
print("\n Getting some statistical data about the dataset")
print(diabetes_data.describe())

print("\n Number of non-diabetic and diabetic cases-- 0->non diabetic 1->diabetic")
print(diabetes_data["Outcome"].value_counts())

#seperating the features and target columns
features=diabetes_data.drop(columns='Outcome',axis=1)
target=diabetes_data['Outcome']

#Data Standardisation
scaler=StandardScaler()
scaler.fit(features)
Standardised_data=scaler.transform(features)

#redefining the features and targets again
features=Standardised_data
target=diabetes_data['Outcome']

#train test splitting
X_train,X_test,Y_train,Y_test=train_test_split(features,target,test_size=0.2,random_state=2)

#training the model
classifier=svm_classifier(learning_rate=0.001,no_of_iterations=1000,lambda_parameter=0.01)

#training the SVM classifier with training data
classifier.fit(X_train,Y_train)

#Model Evaluation
#accuracy on training data
#X_train_prediction has all the values predicted by our model
X_train_prediction=classifier.predict(X_train)
train_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print("\n Accuracy score on training data-",train_data_accuracy)

#accuracy on training data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print("\n Accuracy score on testing data-",test_data_accuracy)

#Building the Prediction System

input_data=(4,110,92,0,0,37.6,0.191,30)
#change the input data into numpy array
numpy_input_data=np.asarray(input_data)
#reshape the array
reshaped_input_data=numpy_input_data.reshape(1,-1)
#standardising the input data
std_input_data=scaler.transform(reshaped_input_data)
predict=classifier.predict(std_input_data)
print("\n Predicted Data-> ",predict)
#predict here is a list with just one element so we take predict[0]
if(predict[0]==1):
    print("\n The persion is Diabetict")
else:
    print("\n The person is Not Diabetic")



