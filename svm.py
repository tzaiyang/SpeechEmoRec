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

def get_input_fetures(features_file,label_file):
    labels = utils.load_labels(label_file, data_root)
    X = np.load(features_file)
    y = np.array(labels, dtype=np.uint8)

    return X,y


if __name__ == '__main__':
    DataDir = path.DataDir
    train_utterance_file = DataDir.train_utterance
    train_data_path = DataDir.train_path
    
    test_utterance_file = DataDir.test_utterance
    test_data_path = DataDir.val_path
   
    data_root = DataDir.DataRoot
    svm_model_file = DataDir.svm

    #for i in range(0,len(DataDir.val_speaker)):
    for i in range(0,10):
        X_train,y_train = get_input_fetures(train_utterance_file[i],train_data_path[i])
        X_test,y_test = get_input_fetures(test_utterance_file[i],test_data_path[i])
        print("input shape:")
        print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
        print("starting training...")
        clf = SVC(probability=True)
        clf.fit(X_train, y_train)
    
        print("speaker %s train dataset accuracy: %10.8s"%(DataDir.val_speaker[i],clf.score(X_train, y_train)))
        print("speaker %s test dataset accuracy: %10.8s"%(DataDir.val_speaker[i],clf.score(X_test, y_test)))
    
        joblib.dump(clf, svm_model_file[i])
