import numpy as np
from sklearn.svm import SVC
import utils
from sklearn.externals import joblib

from global_path import Data_Directory

test_data_path = '%s%s' % (Data_Directory, 'txts/03/test_data.txt')
train_data_path = '%s%s' % (Data_Directory, 'txts/03/train_data.txt')
data_root = '%s%s' % (Data_Directory, 'melSpec_Bolin_Speech')

train_paths, train_labels = utils.load_paths(train_data_path, data_root)
test_paths, test_labels = utils.load_paths(test_data_path, data_root)

train_feature_file = '%s%s' % (Data_Directory, 'train_features.npy')
X_train = np.load(train_feature_file)
y_train = np.array(train_labels, dtype=np.uint8)

test_feature_file = '%s%s' % (Data_Directory, 'test_features.npy')
X_test = np.load(test_feature_file)
y_test = np.array(test_labels, dtype=np.uint8)

clf = SVC(probability=True)
clf.fit(X_train, y_train)

num_test = 10

print('training phrase')

print(clf.predict([X_train[i] for i in range(num_test)]))
print([y_train[i] for i in range(num_test)])

print(clf.score(X_train, y_train))

print('testing phrase')
print(clf.predict([X_test[i] for i in range(num_test)]))
print([y_test[i] for i in range(num_test)])

print(clf.score(X_test, y_test))

svm_model_file = '%s%s' % (Data_Directory, 'svm_model.m')
joblib.dump(clf, svm_model_file)

# clf = joblib.load('Cry_train_model.m')
# print(clf.predict(X_test))
#
# print(y_test)
#
# print(clf.predict_proba(X_test))
