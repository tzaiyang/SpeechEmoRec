import numpy as np
from sklearn.svm import SVC
import utils
from sklearn.externals import joblib
import path

def get_pred_labes(svm_model_file,test_utterance_file):
    X_test = np.load(test_utterance_file)
    clf = joblib.load(svm_model_file)
    
    y_pred = clf.predict(X_test)
    
    #y_pred_proba = clf.predict_proba(X_test)
    # print(y_pred)
    # print(y_pred_proba)

    return y_pred

if __name__ == '__main__':
    DataDir = path.DataDir
    train_utterance_file = DataDir.train_utterance
    test_utterance_file = DataDir.test_utterance
    train_data_path = DataDir.train_path
    test_data_path = DataDir.val_path
    data_root = DataDir.DataRoot
    svm_model_file = DataDir.svm

    num_test = 10

    train_paths, train_labels = utils.load_paths(train_data_path, data_root)
    test_paths, test_labels = utils.load_paths(test_data_path, data_root)

    X_train = np.load(train_utterance_file)
    y_train = np.array(train_labels, dtype=np.uint8)

    X_test = np.load(test_utterance_file)
    y_test = np.array(test_labels, dtype=np.uint8)

    print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    clf = SVC(probability=True)
    clf.fit(X_train, y_train)

    print('training phrase')
    #print(clf.predict([X_train[i] for i in range(num_test)]))
    #print([y_train[i] for i in range(num_test)])
    print(clf.score(X_train, y_train))

    print('testing phrase')
    #print(clf.predict([X_test[i] for i in range(num_test)]))
    #print([y_test[i] for i in range(num_test)])
    print(clf.score(X_test, y_test))

    joblib.dump(clf, svm_model_file)
