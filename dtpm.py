import numpy as np
import sklearn.discriminant_analysis as lda
from sklearn.neighbors import NearestNeighbors
import sys
import get_fc7
import utils
import tensorflow as tf

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

    features_Vp = np.concatenate((L0, L1_1, L1_2, L2_1, L2_2, L2_3, L2_4), axis=0)

    features_Up = np.matmul(Weights_L,features_Vp)

    return features_Up

def lpnorm_pooling(features_Ln,var_p):
    '''
    :param features_Ln:
    :param var_p: 1-average pooling, np.inf-max pooling
    :return:
    '''
    lpnorm = np.linalg.norm(features_Ln,ord=var_p)
    result = lpnorm * (1/features_Ln.shape[0])**var_p

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


def solve_weights(graph_filename,load_filename):
    features_Vp, labels = get_features_Vp(graph_filename, load_filename)
    clf = lda.LinearDiscriminantAnalysis(solver='eigen',shrinkage=None,priors=None,
                                         n_components=None)
    clf.fit(features_Vp, labels)
    print(clf.predict(features_Vp[0]))
    print(clf.coef_)
    return clf.coef_

def get_object_function(graph_filename,load_filename):
    features_Vp, labels = get_features_Vp(graph_filename, load_filename)
    weights = solve_weights(graph_filename,load_filename)
    nbrs = NearestNeighbors(n_neighbors=20)
    Sb = []
    Sw = []
    for i in range(0,features_Vp.shape[0]):
        for j in range(0,20):
            Upnearest_same = nbrs.kneighbors(features_Vp[i], neighbors=20, return_distance=False)
            Upnearest_diff = nbrs.kneighbors(features_Vp[i], neighbors=20, return_distance=False)

            Sb += np.matmul((features_Vp[i]- Upnearest_diff[j]),np.transpose(features_Vp[i]- Upnearest_diff[j]))
            Sw += np.matmul((features_Vp[i]- Upnearest_same[j]),np.transpose(features_Vp[i]- Upnearest_same[j]))


    object_fuction_numerator = np.matmul(np.matmul(np.transpose(weights),Sb),weights)
    object_fuction_denominator = np.matmul(np.matmul(np.transpose(weights),Sw),weights)
    object_fuction = np.divide(object_fuction_numerator,object_fuction_denominator)

    return object_fuction

def solve_p(learning_rate,Niter,var_p):
    object_function = - get_object_function(graph_filename,load_filename)
    gradients = tf.gradients(object_function, var_p)
    gradients = list(zip(gradients, var_p))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    with tf.Session as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        for i in range(Niter):
            sess.run(train_op)


def get_features_Vp(graph_filename,load_filename):
    features,labels= get_fc7.get_fc7(graph_filename,load_filename)
    features_Vp=[]
    for i in range(len(features)):
        feat = dtpm(features[i])
        features_Vp.append(feat)
    features_Vp = np.asarray(features_Vp, dtype=np.float32)
    return features_Vp,labels

def save_features_Vp(graph_filename,load_filename,save_filename):
    features_Vp,labels= get_features_Vp(graph_filename, load_filename)
    print(features_Vp.shape)
    np.save(save_filename, features_Vp)

if __name__ == '__main__':

    graph_filename = 'alexnet.pb'
    data_root = './'
    learning_rate = 0.001

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

    save_features_Vp(graph_filename, load_filename, save_filename)

    solve_weights(graph_filename, load_filename)