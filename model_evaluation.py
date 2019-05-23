import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

dataRead = pandas.read_csv("./data.csv")
dataframe = dataRead.drop(['SubMatches', 'SubKeypoint'], axis = 1)

array = dataframe.values
X = array[:,0:4]
Y = array[:,4]
test_size = 0.1
seed = 7
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size = test_size, random_state = seed)

print('====================X - y====================')
print('X_train: ', X_train[0:20])
print('X_test: ', X_test[0:20])


model = LogisticRegression()
model.fit(X_train, y_train)
# save the model to disk
#filename = 'myModel_and_value.sav'
#pickle.dump(model, open(filename, 'wb'))

y_pred = model.predict(X_test)
y_true = y_test

print('======================Confusion matrix====================================')
confusionMatrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix: ')
print(confusionMatrix)
accuracy = np.diagonal(confusionMatrix).sum()/confusionMatrix.sum()
print('Accuracy = ', accuracy)
normalizedConfusionMatrix = confusionMatrix/confusionMatrix.sum(axis = 1, keepdims = True)
print('Normalized Confusion Matrix: ')
print(normalizedConfusionMatrix)


print('=======================Precision - Recall=================================')

print('=======Class 1========')
p1 = confusionMatrix[1,1]/np.sum(confusionMatrix[:,1])
r1 = confusionMatrix[1,1]/np.sum(confusionMatrix[1])
f1_1 = (2*p1*r1)/(p1+r1)
print('precision label 1 = ', p1)
print('recall lable 1 = ', r1)
print('f1-score label 1 = ', f1_1)

print('=======Class 0========')
p0 = confusionMatrix[0,0]/np.sum(confusionMatrix[:,0])
r0 = confusionMatrix[0,0]/np.sum(confusionMatrix[0])
f1_0 = (2*p0*r0)/(p0+r0)
print('precision label 0 = ', p0)
print('recall lable 0 = ', r0)
print('f1-score label 1 = ', f1_0)


print('================ababababababababababbaab====================')
#print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(confusionMatrix, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(confusionMatrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()
