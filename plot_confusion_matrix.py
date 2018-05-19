"""
================
Confusion matrix
================
"""
# print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import path
import svm
import utils

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100 
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




if __name__ == '__main__':
    DataDir = path.DataDir
    train_utterance_file = DataDir.train_utterance
    test_utterance_file = DataDir.test_utterance
    train_data_path = DataDir.train_path
    svm_model_file = DataDir.svm
    
    test_data_path = DataDir.val_path
    data_root = DataDir.DataRoot
    confusion_matrix_file = DataDir.confusion_matrix
    
    class_names = ['anger','boredom','disgust','fear','happiness','sadness','neutral']
    cnf_matrix = np.zeros((7,7))
    #for i in range(0,len(DataDir.val_speaker)):
    for i in [0,1,5,6,7]:#range(0,10):
        y_true = utils.load_labels(test_data_path[i],data_root)
        y_pred = svm.get_pred_labes(svm_model_file[i],test_utterance_file[i])
        
        print(y_true)
        print(y_pred)
        # Compute confusion matrix
        cnf_matrix += confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
    
        # Plot non-normalized confusion matrix
        #plt.figure()
        #plot_confusion_matrix(cnf_matrix, classes=class_names,
        #                    title='Confusion matrix, without normalization')
        #plt.savefig('xx2.png')
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')
        plt.savefig(confusion_matrix_file[i])
        plt.show()
