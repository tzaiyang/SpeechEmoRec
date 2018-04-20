import numpy as np,wave
import scipy as sp
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
from pylab import *

Dataset_path = "./EMODB"
Dataset_DCNN = "./DCNN_IN"
Dataset_DEBUG = "./DEBUG"

# if you want to see utterance mel spectrogram and delta,delta-delta picture,
# set __DEBUG_ as True,and the pictures will be DEBUG directory
_DEBUG_ = True

def dir_prepare():
    if os.path.isdir(Dataset_path) == False:
        os.makedirs(Dataset_path)
    if os.path.isdir(Dataset_DCNN) == False:
        os.makedirs(Dataset_DCNN)
    if os.path.isdir(Dataset_DEBUG) == False:
        os.makedirs(Dataset_DEBUG)

    wave_filenames = os.listdir(Dataset_path)
    for filename in wave_filenames:
        if os.path.isdir('%s/%s/%s' % (Dataset_DCNN, filename[:2], filename[2:])) == False:
            os.makedirs('%s/%s/%s' % (Dataset_DCNN, filename[:2], filename[2:]))

def classfier_dataset_todir():
    wave_filenames = os.listdir(Dataset_path)
    for filename in wave_filenames:
        if os.path.isdir('%s/%s/%s'%(Dataset_path,filename[:2],filename[2:5])) == False:
            os.makedirs('%s/%s/%s'%(Dataset_path,filename[:2],filename[2:5]))
        os.rename('%s/%s'%(Dataset_path,filename),'%s/%s/%s/%s'%(Dataset_path,filename[:2],filename[2:5],filename[5:12]))

def read_wav(wav_path):
    wavefile = wave.open(wav_path)
    nchannels,sampwidth,framerate,nframes,comptype,compname=wavefile.getparams()
    strdata = wavefile.readframes(nframes)
    wavedata = np.fromstring(strdata, dtype=np.int16).astype('float32')# / (2 ** 15)
    print ('nchnnels:%d'%nchannels)
    print ('sampwidth:%d'%sampwidth)
    print ('framerate:%d'%framerate)
    print ('nframes:%d'%nframes)
    #print ('nframes: ')
    #print (wavedata)
    wavefile.close()
    return nchannels,sampwidth,framerate,nframes,wavedata

def gen_utterance_melspec(wav_path):
    """
    Compute a mel-scaled spectrogram to a utterance wavefile
    :param wav_path: audio time-series file
    :return:
    """
    nchannels,sampwidth,framerate,nframes,wavedata = read_wav(wav_path)
    #wavedata,framerate = librosa.core.load(wav_path)

    # method 1:hamming window
    #signal.get_window('hamming', 7)
    #Zxx = librosa.core.stft(wavedata, n_fft=25*framerate/1000, hop_length=(25-10)*framerate/1000, window='hamming', center=True, pad_mode='reflect')
    #Sxx = librosa.feature.melspectrogram(S=np.abs(Zxx),n_mels=64, fmin=20, fmax=8000)

    # method 2:hanning window
    Sxx = librosa.feature.melspectrogram(y=wavedata,sr=framerate,n_fft=25*framerate/1000,hop_length=(25-10)*framerate/1000,n_mels=64,fmin=20,fmax=8000)
    return Sxx

def save_utterance(X,savepath,filename="utterance"):
    # Convert a power spectrogram (amplitude squared) to decibel (dB) units
    X = librosa.power_to_db(X, ref=np.max)
    # Display a spectrogram/chromagram/cqt/etc.
    librosa.display.specshow(X,fmin=20, fmax=8000)
    plt.savefig("%s/%s"%(savepath,filename),bbox_inches='tight',pad_inches=0)

def gen_segments_melspec(X, window_size, overlap_sz):
    """
    Create an overlapped version of X

    Parameters
    ----------
    X : ndarray, shape=(n_mels,n_samples)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    overlap_sz : int
        Step size between windows

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    window_step = (window_size-overlap_sz)
    append = np.zeros((64,(window_step - (X.shape[-1]-overlap_sz) % window_step)))
    X = np.hstack((X, append))
    # append zeros of end of X to get integer numbers of n_windows
    new_shape = ((X.shape[-1] - overlap_sz) // window_step,window_size,X.shape[0])
    new_strides = (window_step*8,X.strides[0],X.strides[-1])
    X_strided = np.lib.stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)

    return X_strided

def save_segment(X,savepath,filename='static',num=0):
        # librosa.display.specshow(X, fmin=20, fmax=8000)
        # plt.savefig("%s/mel_%s%d.png"%(savepath,filename,num),bbox_inches='tight',pad_inches=0)


        # librosa.display.specshow(delta, fmin=20, fmax=8000)
        # plt.savefig("mel_%s%d.png" % ('delta', num), bbox_inches='tight', pad_inches=0)
        #
        # librosa.display.specshow(delta_delta, fmin=20, fmax=8000)
        # plt.savefig("mel_%s%d.png" % ('delta_delta', num), bbox_inches='tight', pad_inches=0)


        fig = plt.figure(frameon=False)
        fig.set_size_inches(6, 6)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(X, aspect='equal')
        fig.savefig("%s/mel_%s%d.png"%(savepath,filename,num), dpi=100)

        # plt.imshow(librosa.power_to_db(X[num], ref=np.max), aspect='equal')
        # show()
        # plt.savefig("mel_%s%d.png"%(name,num), dpi=100)

def save_dcnn_ipput(X,num,savepath,filename):
    # librosa.display.specshow(static, fmin=20, fmax=8000)
    # plt.savefig("mel_%s%d.png" % ('static', i), bbox_inches='tight', pad_inches=0)

    # plt.imshow(X)
    # # #show()
    # # plt.imsave('images%d.png'%i,images)
    # plt.savefig('%s/%s%d.png' % (savepath, filename, num))

    fig = plt.figure(frameon=False)
    fig.set_size_inches(6, 6)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(X, aspect='equal')
    fig.savefig('%s/%d.png' % (savepath, num))
    close()


def gen_dcnn_input(wav_path,savepath,filename):
    utterance_melspec = gen_utterance_melspec(wav_path)
    if _DEBUG_:
        save_utterance(utterance_melspec,Dataset_DEBUG)

    segments_melspec = gen_segments_melspec(utterance_melspec,window_size=64,overlap_sz=30)
    for num in range(0, segments_melspec.shape[0]):
        static = librosa.power_to_db(segments_melspec[num], ref=np.max)
        delta = librosa.feature.delta(static, order=1)
        delta_delta = librosa.feature.delta(static, order=2)
        if _DEBUG_:
            save_segment(static,Dataset_DEBUG,filename="static",num=num)
            save_segment(delta, Dataset_DEBUG,filename="delta", num=num)
            save_segment(delta_delta,Dataset_DEBUG,filename="delta_delta", num=num)

        images = np.dstack((static,delta,delta_delta))

        save_dcnn_ipput(images,num,savepath,filename)

if __name__ == '__main__':
    dir_prepare()
    wave_filenames = os.listdir(Dataset_path)
    if _DEBUG_:
        wav_path = '%s/%s' % (Dataset_path, wave_filenames[0])
        gen_dcnn_input(wav_path, Dataset_DEBUG, wave_filenames[0])
        print (wave_filenames[0])
    else:
        for filename in wave_filenames:
            wav_path = '%s/%s' % (Dataset_path,filename)
            save_path = '%s/%s/%s' % (Dataset_DCNN,filename[:2],filename[2:])
            gen_dcnn_input(wav_path,save_path,filename[2:])



