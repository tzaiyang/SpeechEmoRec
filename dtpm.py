import numpy as np
import sklearn.discriminant_analysis as lda
import sys
import get_fc7
import utils

def dtpm(features):
    '''
    :param features: array:shape(N,4096)
    :return:
    '''
    n, d = features.shape
    if n == 3:
        features = np.row_stack((features, features[-1]))
    if n == 2:
        features = np.row_stack((features, features))
    if n == 1:
        features = np.row_stack((features, features, features, features))
    n, d = features.shape

    [a, b], [c, d, e, f] = div_L0(n)

    L0 = lpnorm_pooling(features)

    L1_1 = lpnorm_pooling(features[:a])
    L1_2 = lpnorm_pooling(features[a:])

    L2_1 = lpnorm_pooling(features[:c])
    L2_2 = lpnorm_pooling(features[c:a])
    L2_3 = lpnorm_pooling(features[a:a+e])
    L2_4 = lpnorm_pooling(features[a+e:])

    W_L0=1/4;
    W_L1=1/4;
    W_L2=1/2;

    Weights_L = [[W_L0,0,0,0,0,0,0],
                 [0,W_L1,0,0,0,0,0],
                 [0,0,W_L1,0,0,0,0],
                 [0,0,0,W_L2,0,0,0],
                 [0,0,0,0,W_L2,0,0],
                 [0,0,0,0,0,W_L2,0],
                 [0,0,0,0,0,0,W_L2]]

    features_Vp = np.concatenate((W_L0*L0, W_L1*L1_1, W_L1*L1_2, W_L2*L2_1, W_L2*L2_2, W_L2*L2_3, W_L2*L2_4), axis=0)

    #features_Up = np.matmul(Weights_L,features_Vp)

    return features_Vp

def lpnorm_pooling(features_Ln):
    result = np.max(features_Ln,axis=0)
    return result

def div_L0(num):
    [a, b] = div_L1(num)
    [c, d, e, f] = div_L2(num)

    return [a, b], [c, d, e, f]


def div_L1(num):
    a = num // 2
    b = num - a

    return [a, b]


def div_L2(num):
    [a, b] = div_L1(num)
    [c, d] = div_L1(a)
    [e, f] = div_L1(b)

    return [c, d, e, f]


def Solve_Weights(features_Up):
    X = features_Up
    y = np.array([1, 1, 1, 2, 2, 2])

    clf = lda.LinearDiscriminantAnalysis(solver='eigen',shrinkage=None,priors=None,
                                         n_components=None)
    clf.fit(X, y)
    print(clf.predict([[-0.8, -1]]))
    print(clf.coef_)


if __name__ == '__main__':

    graph_filename = 'alexnet.pb'
    data_root = './'

    if len(sys.argv) > 1:
        if sys.argv[1] == '-t':
            save_filename= 'train_features.npy'
            load_filename= 'Dataset/train.txt'
            print ('save train_fetures')
        elif sys.argv[1] == '-v':
            save_filename = 'test_features.npy'
            load_filename = 'Dataset/val.txt'
            print ('save test_fetures')
        else:
            print('please input the arguments,e.g.\n python dtpm.py -t\n python dtpm.py -v')
            exit(0)
    else:
        print('please input the arguments,e.g.\n python dtpm.py -t\n python dtpm.py -v')
        exit(0)

    features = get_fc7.get_fc7(graph_filename,load_filename)
    features_Up=[]
    for i in range(len(features)):
        feat = dtpm(features[i])
        features_Up.append(feat)

    features_Up = np.asarray(features_Up, dtype=np.float32)
    print(features_Up.shape)
    np.save(save_filename, features_Up)
