import pandas
import numpy
import numpy as np
import warnings
import itertools
import matplotlib.pyplot as plt
import seaborn
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics as metrics
from keras.models import Sequential
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Conv1D , Dropout, Activation
from keras.layers import LSTM,Dense, Flatten
from keras.callbacks import ModelCheckpoint, History
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings("ignore")

train = pandas.read_csv("traindata.csv")
test = pandas.read_csv("testdata.csv")

print(train.head())

print("Training data has {} rows & {} columns".format(train.shape[0],train.shape[1]))

print(test.head())

print("Testing data has {} rows & {} columns".format(test.shape[0],test.shape[1]))

train.describe()

ratio = train['class'].value_counts()
labels = ratio.index[0], ratio.index[1]
sizes = [ratio.values[0], ratio.values[1]]

figure, axis = plt.subplots()
axis.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axis.axis('equal')

plt.show()

print(train['num_outbound_cmds'].value_counts())
print(test['num_outbound_cmds'].value_counts())

train.drop(['num_outbound_cmds'], axis=1, inplace=True)
test.drop(['num_outbound_cmds'], axis=1, inplace=True)

scaler = StandardScaler()

cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))

sc_traindf = pandas.DataFrame(sc_train, columns = cols)
sc_testdf = pandas.DataFrame(sc_test, columns = cols)

encoder = LabelEncoder()

cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()

traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)

enctrain = traincat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()

train_x = pandas.concat([sc_traindf,enctrain],axis=1)
train['class']=le.fit_transform(train['class'])
train_y = train['class']
train_x.shape

test_df = pandas.concat([sc_testdf,testcat],axis=1)
test_df.shape

rfc = RandomForestClassifier();

rfc.fit(train_x, train_y);

score = numpy.round(rfc.feature_importances_,3)
importances = pandas.DataFrame({'feature':train_x.columns,'importance':score})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

plt.rcParams['figure.figsize'] = (16,4)
importances.plot.bar();

rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(train_x, train_y)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_x.columns)]
selected_features = [v for i, v in feature_map if i==True]

print(selected_features)

seaborn.heatmap(train_x[selected_features].corr(), annot = True, fmt='.1g')

X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,train_size=0.60, random_state=2)

model = KNeighborsClassifier(n_jobs=-1)
model.fit(X_train, Y_train);

scores = cross_val_score(model, X_train, Y_train, cv=10)
accuracy = metrics.accuracy_score(Y_train, model.predict(X_train))
confusion_matrix = metrics.confusion_matrix(Y_train, model.predict(X_train))
classification = metrics.classification_report(Y_train, model.predict(X_train))

print ("Cross Validation Mean Score:" "\n", scores.mean())
print ("Model Accuracy:" "\n", accuracy)
print ("Confusion matrix:" "\n", confusion_matrix)
print ("Classification report:" "\n", classification)

accuracy = metrics.accuracy_score(Y_test, model.predict(X_test))
confusion_matrix = metrics.confusion_matrix(Y_test, model.predict(X_test))
classification = metrics.classification_report(Y_test, model.predict(X_test))
                                                                     
print ("Model Accuracy:" "\n", accuracy)
print ("Confusion matrix:" "\n", confusion_matrix)
print ("Classification report:" "\n", classification)

prediction = model.predict(test_df)
test['prediction'] = prediction
print(test.head())

prediction

ratio = test['prediction'].value_counts()
labels = ratio.index[0], ratio.index[1]
sizes = [ratio.values[0], ratio.values[1]]

figure, axis = plt.subplots()
axis.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axis.axis('equal')

plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
train_X = scaler.transform(X_train)
test_X = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(train_X,Y_train)

y_pred=mlp.predict(test_X)
y_pred
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix,accuracy_score
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
from tensorflow.keras.utils import to_categorical

y_train_binary = to_categorical(Y_train,5)

y_test_binary = to_categorical(Y_test,5)

x_train = train_X.reshape(X_train.shape[0],X_test.shape[1],1)
x_test = test_X.reshape(X_test.shape[0], X_test.shape[1],1)

print("\n ****************** CNN ******************** \n")
##
num_classes = 5
input_shape = (20,1)
batch_size = 3500
epochs = 20
model_cnn = Sequential()
model_cnn.add(Conv1D(32, (3), input_shape=(x_train.shape[1],1), activation='relu'))
model_cnn.add(Flatten())
model_cnn.add(Dense(64, activation='relu'))
model_cnn.add(Dense(num_classes, activation='sigmoid'))

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_cnn.summary()

history=model_cnn.fit(x_train, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_binary))

##
y_pred_cnn=np.argmax(model_cnn.predict(x_test), axis=-1)

print(np.argmax(y_pred_cnn,axis=0))

print(classification_report(Y_test, y_pred_cnn))
print('Accuracy of CNN: ', accuracy_score(Y_test, y_pred_cnn))
#plot. loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training

plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix

import scikitplot.plotters as skplt
##
def plot_cmat(y_test, y_pred_cnn):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(Y_test, y_pred_cnn)
    plt.show()
plot_cmat(Y_test, y_pred_cnn)

print("\n ****************** RNN ******************** \n")
num_classes = 5
input_shape = (20,1)
batch_size = 3600
epochs = 100
model = Sequential()
model.add(LSTM(units = 10,
                   input_shape=(x_train.shape[1],1),
                   activation = 'sigmoid',
                   return_sequences=True))
model.add(Flatten())
model.add(Dense(64, activation='softmax'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history=model.fit(x_train, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_binary))
y_pred1=np.argmax(model_cnn.predict(x_test), axis=-1)

print(np.argmax(y_pred1,axis=0))


plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training

plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
print('Accuracy of LSTM: ', accuracy_score(Y_test, y_pred1))
print(classification_report(Y_test, y_pred1))
