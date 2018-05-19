from __future__ import division, print_function, absolute_import
import os

#EMODB_ROOT= 'EMODB/'
EMODB_ROOT= '/home/datasets/tzaiyang/EMODB/'
ENTERFACE_ROOT = '/home/datasets/public/eNTERFACE/'
BAULMS_ROOT='BAULMS/'

emodb_speaker = ['03','08','09','10','11','12','13','14','15','16']
enterface_nspeaker = 44 + 1
enterface_speaker = map(str,range(1,enterface_nspeaker))

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
        self.wav=root_dir + "wav"
        self.DCNN_IN=root_dir + "DCNN_IN"
        self.DEBUG=root_dir +  "DEBUG"
        
        if self.DataRoot== EMODB_ROOT:
            self.val_speaker = emodb_speaker
            self.labels_dict = emodb_labels_dict
        elif self.DataRoot== ENTERFACE_ROOT:
            self.val_speaker = enterface_speaker
            self.labels_dict = enterface_labels_dict
        
        self.nclasses = len(self.labels_dict)
        self.define_filename()


    def define_filename(self):
        #speakerhome is speaker number index
        self.train_segments_path = []
        self.val_segments_path=[]
        self.train_path=[]
        self.val_path=[]

        self.train_utterance=[]
        self.test_utterance =[]
        self.train_segments=[]
        self.test_segments=[]
        
        self.alexnet=[]
        self.svm =[]
        self.confusion_matrix =[]
        
        for i in range(len(self.val_speaker)):
            speakerhome = self.DataRoot + self.val_speaker[i] + '/'
            if not os.path.exists(speakerhome):
                os.makedirs(speakerhome)
            
            self.train_segments_path.append(speakerhome + "train_segments.txt")
            self.val_segments_path.append(speakerhome +  "val_segments.txt")
            self.train_path.append(speakerhome +  "train.txt")
            self.val_path.append(speakerhome +  "val.txt")
            
            self.train_utterance.append(speakerhome + 'train_utterance.npy')
            self.test_utterance.append(speakerhome + 'test_utterance.npy')
            self.train_segments.append(speakerhome + 'train_segments.npy')
            self.test_segments.append(speakerhome + 'test_segments.npy')
            
            self.alexnet.append(speakerhome + "alexnet.pb")
            self.svm.append(speakerhome +  'svm_model.m')
            self.confusion_matrix.append(speakerhome +  'confusion_matrix')
        


    def percent_bar(self,numerator,denominator):
        percent = 1.0 * numerator/ denominator * 100
        print ("complete precent:%10.8s%s"%(percent,'%'),end='\r')
        if numerator == denominator :
            print ("%d items was processed successfully"%numerator) 

    def delete_pathfile(self):
        for i in range(len(self.val_speaker)):
            if os.path.exists(self.train_path[i]):
                os.remove(self.train_path[i])
            if os.path.exists(self.val_path[i]):
                os.remove(self.val_path[i])
            if os.path.exists(self.train_segments_path[i]):
                os.remove(self.train_segments_path[i])
            if os.path.exists(self.val_segments_path[i]):
                os.remove(self.val_segments_path[i])

    def stat_labels_pos(self,filename):
        if self.DataRoot == EMODB_ROOT:
           self.labels_index = filename[5:6] 
        elif self.DataRoot == ENTERFACE_ROOT:
           self.labels_index = filename.split('_')[-2]
   
    def stat_speaker_pos(self,wav_path):
        if self.DataRoot == EMODB_ROOT:
            return wav_path.split('/')[-1][:2]
        elif self.DataRoot == ENTERFACE_ROOT:
            return wav_path.split('/')[-1].split('_')[0].split(']')[-1][1:]
        
    def split_val_set(self,wav_path,savepath,train_fname,val_fname):
        for i in range(len(train_fname)):
            train_file = open(train_fname[i],'a')
            val_file = open(val_fname[i],'a')
    
            if self.stat_speaker_pos(wav_path)  == self.val_speaker[i]:
                val_file.write('%s %s\n' % (savepath, self.labels_dict[self.labels_index]))
            else:
                train_file.write('%s %s\n' % (savepath, self.labels_dict[self.labels_index]))
    
            train_file.close()
            val_file.close()


DataDir = DatasetDir(EMODB_ROOT)

