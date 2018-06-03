from sklearn.externals import joblib
import numpy as np
import path
import get_fc7
import dtpm
import melSpec
import utils
import os 

DataDir = path.DataDir
svm_model_file = DataDir.svm
graph_filename = DataDir.alexnet
test_filename = DataDir.val_path
weights = np.array([1])

filename = '08a07Na.wav'
wav_path = '%s/%s' % (DataDir.wav, filename)
savepath = 'model_test/'

#MFSC
melSpec.wav_to_pics(wav_path,savepath,1)
# #DCNN
graph_filename = DataDir.alexnet
test_features = get_fc7.get_dcnn_out(graph_filename[0],savepath)
# #DTPM
features_Vp = dtpm.dtpm(test_features)
print(features_Vp.shape,weights.shape[0])

if weights.shape[0] == 1:
    features_Up=features_Vp
    # print('Up=Vp')
else:
    features_Up = np.multiply(features_Vp,weights)
    print(weights.shape[0])
    print(features_Up.shape)
    # print('Up=W*Vp')

#SVM predicting
class_names = ['anger','boredom','disgust','fear','happiness','sadness','neutral']
clf = joblib.load(svm_model_file[0])
y_pred = clf.predict([features_Up])
labels = class_names[int(path.emodb_labels_dict[filename[5:6]])]
print('\nLabels:%s'%labels)
print('Predicting result:%s'%class_names[int(y_pred)])