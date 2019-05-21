import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
filePath = "./data_processed.csv"
dataRead = pandas.read_csv(filePath)

dataframe = dataRead.drop(['SubMatches', 'SubKeypoint'], axis=1)
#print(dataframe)

array = dataframe.values
X = array[:,0:4]
Y = array[:,4]
#print('X:::::::::', X)
#print('Y:::::::::', Y)
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk

filename = 'myModel94.sav'
pickle.dump(model, open(filename, 'wb'))


#
## some time later...
#
## load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)
