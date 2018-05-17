from __future__ import division, print_function, absolute_import
import os

EMODB_ROOT= 'EMODB/'
ENTERFACE_ROOT = '/home/datasets/public/eNTERFACE/'
BAULMS_ROOT='BAULMS/'

#
# labels_dict = {'W':[1,0,0,0,0,0,0],
#                'L':[0,1,0,0,0,0,0],
#                'E':[0,0,1,0,0,0,0],
#                'A':[0,0,0,1,0,0,0],
#                'F':[0,0,0,0,1,0,0],
#                'T':[0,0,0,0,0,1,0],
#                'N':[0,0,0,0,0,0,1]}

emodb_labels_dict = {'W':0, # anger
               'L':1, # boredom
               'E':2, # disgust
               'A':3, # fear/anxiety
               'F':4, # happiness
               'T':5, # sadness
               'N':6  # neutral
}
enterface_labels_dict = {'sa':0,
                         'fe':1,
                         'an':2,
                         'di':3,
                         'ha':4,
                         'su':5
                        }                         
class DatasetDir:
    def __init__(self,root_dir):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.DataRoot = root_dir
        self.wav= root_dir+"wav"
        self.DCNN_IN= root_dir+"DCNN_IN"
        self.DEBUG= root_dir+"DEBUG"
        self.train_segments_path= root_dir+"train_segments.txt"
        self.val_segments_path= root_dir+"val_segments.txt"
        self.train_path= root_dir+"train.txt"
        self.val_path= root_dir+"val.txt"

        self.train_utterance=root_dir+'train_utterance.npy'
        self.test_utterance =root_dir+'test_utterance.npy'
        self.train_segments=root_dir+'train_segments.npy'
        self.test_segments=root_dir+'test_segments.npy'
        
        self.alexnet=root_dir+"alexnet.pb"
        self.svm = root_dir+'svm_model.m'

        if root_dir == EMODB_ROOT:
            self.val_speaker = '09'
            self.labels_dict = emodb_labels_dict
        elif root_dir == ENTERFACE_ROOT:
            self.val_speaker = 's15'
            self.labels_dict = enterface_labels_dict
        
        self.nclasses = len(self.labels_dict)

    def percent_bar(self,numerator,denominator):
        percent = 1.0 * numerator/ denominator * 100
        print ("complete precent:%10.8s%s"%(percent,'%'),end='\r')
        if numerator == denominator :
            print ("%d items was processed successfully"%numerator) 

    def delete_pathfile(self):
        if os.path.exists(self.train_path):
            os.remove(self.train_path)
        if os.path.exists(self.val_path):
            os.remove(self.val_path)
        if os.path.exists(self.train_segments_path):
            os.remove(self.train_segments_path)
        if os.path.exists(self.val_segments_path):
            os.remove(self.val_segments_path)

    def stat_labels_pos(self,filename):
        if self.DataRoot == EMODB_ROOT:
           self.labels_index = filename[5:6] 
        elif self.DataRoot == ENTERFACE_ROOT:
           self.labels_index = filename.split('_')[-2]
   
    def stat_speaker_pos(self,wav_path):
        if self.DataRoot == EMODB_ROOT:
            return wav_path.split('/')[-1][:2]
        elif self.DataRoot == ENTERFACE_ROOT:
            return wav_path.split('/')[-1].split('_')[0].split(']')[-1]
        
    def split_val_set(self,wav_path,savepath,train_fname,val_fname):
        train_file = open(train_fname,'a')
        val_file = open(val_fname,'a')

        if self.stat_speaker_pos(wav_path)  == self.val_speaker:
            val_file.write('%s %s\n' % (savepath, self.labels_dict[self.labels_index]))
        else:
            train_file.write('%s %s\n' % (savepath, self.labels_dict[self.labels_index]))

        train_file.close()
        val_file.close()


DataDir = DatasetDir(EMODB_ROOT)

