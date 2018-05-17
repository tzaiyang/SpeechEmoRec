import numpy as np
from sklearn.svm import SVC
import utils
from sklearn.externals import joblib
import path

DataDir = path.DataDir
train_utterance_file = DataDir.train_utterance
test_utterance_file = DataDir.test_utterance
train_data_path = DataDir.train_path
test_data_path = DataDir.val_path
data_root = DataDir.DataRoot

train_paths, train_labels = utils.load_paths(train_data_path, data_root)
test_paths, test_labels = utils.load_paths(test_data_path, data_root)

X_train = np.load(train_utterance_file)
y_train = np.array(train_labels, dtype=np.uint8)

X_test = np.load(test_utterance_file)
y_test = np.array(test_labels, dtype=np.uint8)

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
clf = SVC(probability=True)
clf.fit(X_train, y_train)

num_test = 10

print('training phrase')
#print(clf.predict([X_train[i] for i in range(num_test)]))
#print([y_train[i] for i in range(num_test)])
print(clf.score(X_train, y_train))



print('testing phrase')
#print(clf.predict([X_test[i] for i in range(num_test)]))
#print([y_test[i] for i in range(num_test)])
print(clf.score(X_test, y_test))

svm_model_file = 'svm_model.m'
joblib.dump(clf, svm_model_file)

# clf = joblib.load('Cry_train_model.m')
# print(clf.predict(X_test))
#
# print(y_test)
#
# print(clf.predict_proba(X_test))
