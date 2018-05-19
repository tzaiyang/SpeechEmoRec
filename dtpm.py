import numpy as np
import sklearn.discriminant_analysis as lda
from sklearn.neighbors import NearestNeighbors
import sys
import get_fc7
import utils
import tensorflow as tf
import path

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
    '''
    :param features_Ln:
    :param var_p: 1-average pooling, np.inf-max pooling
    :return:
    '''
    var_p = 2.14  # average pooling
#   var_p = np.inf  # max pooling
    lpnorm = np.linalg.norm(features_Ln,ord=var_p,axis=0)
    result = lpnorm * (1/features_Ln.shape[0])**(1/var_p)

    #print(result)
    result = np.max(features_Ln,axis = 0)
    #result = np.average(features_Ln,axis = 0)
    #print(result)

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

def solve_weights(features_Vp,labels):
    clf = lda.LinearDiscriminantAnalysis()#solver='eigen',shrinkage='auto',priors=None,n_components=None)
    clf.fit(features_Vp, labels)
#    print(clf.predict(features_Vp[0]))
    #print(clf.coef_)
    features_Up = clf.transform(features_Vp)
    print(clf.coef_.shape,features_Up.shape)
    return clf.coef_,features_Up

def get_object_function(features):
    features_Vp, labels = get_features_Vp(features)
    weights = solve_weights(features_Vp,labels)
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


def get_features_Vp(features):
    features_Vp=[]
    for i in range(len(features)):
        feat = dtpm(features[i])
        features_Vp.append(feat)
    features_Vp = np.asarray(features_Vp, dtype=np.float32)
    return features_Vp

def save_features_Up(features_Vp,weights,save_filename):
    print(features_Vp.shape,weights.shape[0])
    if weights.shape[0] == 1:
        features_Up=features_Vp
        print('Up=Vp')
    else:
        features_Up = np.multiply(features_Vp,weights)
        print(weights.shape[0])
        print(features_Up.shape)
        print('Up=W*Vp')
        
    np.save(save_filename, features_Up)

if __name__ == '__main__':
    DataDir = path.DataDir
    graph_filename = DataDir.alexnet
    data_root = DataDir.DataRoot
    learning_rate = 0.001

    train_utterance_file = DataDir.train_utterance
    train_features_file = DataDir.train_segments
    train_filename = DataDir.train_path
    
    test_utterance_file = DataDir.test_utterance
    test_features_file = DataDir.test_segments 
    test_filename = DataDir.val_path


    if len(sys.argv) > 1:
        if sys.argv[1] == '-s':
            #for i in range(0,len(DataDir.val_speaker)):
            for i in range(2,9):
                train_features,labels= get_fc7.get_fc7(graph_filename[i],train_filename[i])
                np.save(train_features_file[i], train_features)
                print('save speaker %s train features segments'%DataDir.val_speaker[i])
                test_features,labels= get_fc7.get_fc7(graph_filename[i],test_filename[i])
                np.save(test_features_file[i], test_features)
                print('save speaker %s test features segments'%DataDir.val_speaker[i])

        # without tpm and lp_norm pooling
        elif sys.argv[1] == '-n':
            #for i in range(0,len(DataDir.val_speaker)):
            for i in range(2,9):
                train_features = np.load(train_features_file[i])
                features_Vp = get_features_Vp(train_features)
                weights = np.array([1])
                save_features_Up(features_Vp, weights,train_utterance_file[i])
                print('save speaker %s train features utterance'%DataDir.val_speaker[i])
           
                test_features = np.load(test_features_file[i])
                features_Vp = get_features_Vp(test_features)
                save_features_Up(features_Vp, weights,test_utterance_file[i])
                print('save speaker %s test features utterance'%DataDir.val_speaker[i])
            
        # solve weights
        elif sys.argv[1] == '-w':
            print('solve_w')
            train_features = np.load(train_features_file)
            features_Vp = get_features_Vp(train_features)
            paths,labels = utils.load_paths(train_filename,'./')
            labels = np.asarray(labels)
            print(features_Vp.shape,labels.shape)
            weights,features_Up = solve_weights(features_Vp, labels)
            weights = np.array([1])
            save_features_Up(features_Up,weights,train_utterance_file)


            test_features = np.load(test_features_file)
            features_Vp = get_features_Vp(test_features)
            paths,labels = utils.load_paths(test_filename,'./')
            labels = np.asarray(labels)
            print(features_Vp.shape,labels.shape)
            weights,features_Up = solve_weights(features_Vp, labels)
            weights = np.array([1])
            save_features_Up(features_Up,weights,test_utterance_file)
            
            print('solve_w')
        # without lp_norm pooling

        elif sys.argv[1] == '-p':
            print('solve_p')

        else:
            print('please input the arguments,e.g.\n python dtpm.py -s\n python dtpm.py -n')
            exit(0)


    else:
        print('please input the arguments,e.g.\n python dtpm.py -t\n python dtpm.py -v')
        exit(0)


