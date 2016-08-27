import os
import scipy.io.wavfile as wf
from mfcc_feature import mfcc
from numpy import *
from save_model_to_json import *
from use_svm import *
import pdb
def load_data_X_Y(dir_path_to_dataset):
    print_true = True
    X = []  # array
    Y = []
    for parent, dirnames, filenames in os.walk(dir_path_to_dataset):

        for filename in filenames:
            if '.wav' in filename:
                wav_file_path = os.path.join(parent, filename)
                if 'adam_traba' in wav_file_path:
                    y = 1
                elif 'bartek_bulat' in wav_file_path:
                    y = 2
                elif 'damian_bulat' in wav_file_path:
                    y = 3
                elif 'katarzyna_konieczna' in wav_file_path:
                    y = 4
                elif 'konrad_malawski' in wav_file_path:
                    y = 5
                elif 'szczepan_bulat' in wav_file_path:
                    y = 6

                srate, wav_file_data = wf.read(wav_file_path)
                feat_mfcc_of_music = mfcc(wav_file_data, srate)
                # numpy.arry
                frames_total = len(feat_mfcc_of_music[:, 1])
                Y = Y + [y] * frames_total
                feat_mfcc_of_music_list = feat_mfcc_of_music.tolist()

                for frame in range(frames_total):
                    X = X + [feat_mfcc_of_music_list[frame]]

    return X, Y
def load_data_user_chose(wav_path_to_data):

    X = []

    srate, wav_file_data = wf.read(wav_path_to_data)
    feat_mfcc_of_music = mfcc(wav_file_data, srate)
    frames_total = len(feat_mfcc_of_music[:, 1])

    feat_mfcc_of_music_list = feat_mfcc_of_music.tolist()

    for frame in range(frames_total):
        X = X + [feat_mfcc_of_music_list[frame]]

    return X

def simpleTest(wav='01.wav'):
    dataMatT, labelMatT = load_data_X_Y('samples')
    dataX=load_data_user_chose(wav)
    dataX_test=mat(dataX)
    datMat = mat(dataMatT)
    labelMat = mat(labelMatT).transpose()
    print "begin try to load model"

    try:
        SVMClassifier_simple = objectLoadFromFile('SVMClassifier_simple.json')
        SVMClassifier_simple.jsonLoadTransfer()  # change back to numpy matrix
        print 'load SVMClassifier successfully'
    except IOError, ValueError:

        print 'SVM classifer file doesnt exist, Train first'

        testSvm = svmTrain(datMat, labelMat, 0.6, 0.001)
        b, alpha = testSvm.smoP()
        vecAlp, vec, vecClass = testSvm.svmSurpportVecsGet()
        SVMClassifier_simple = svmClassifer(b, vecAlp, vec, vecClass)
        SVMClassifier_simple.jsonDumps('SVMClassifier_simple.json')  # change to python default list
        # pdb.set_trace()
        SVMClassifier_simple.jsonLoadTransfer()  # change back to numpy matrix
        print "save the model"

    m, n = datMat.shape
    errorCount = 0.0

    for i in range(100):
        dataIn = datMat[i, :]
        print dataIn
        result = SVMClassifier_simple.svmClassify(dataIn)
        print 'predict result is: ', result, ' real result is: ', labelMatT[i]
        if result != labelMatT[i]: errorCount += 1
    print 'errorRate:', errorCount / 100

    data_pridect_Y=SVMClassifier_simple.svmClassify(dataX_test)
    vote=vote_the_max_times(data_pridect_Y)

    speaker_name=get_speaker_name(vote)

    return speaker_name
def vote_the_max_times(data_pridect_Y):
    data_pridect_Y=data_pridect_Y.tolist()
    items = dict([(data_pridect_Y.count(i), i) for i in data_pridect_Y])
    max_times=(int(items[max(items.keys())]))
    return max_times
def get_speaker_name(vote):
    if vote==1:
        pridect_name='adam'
    elif vote==2:
        pridect_name = 'bartek'
    elif vote==3:
        pridect_name = 'damian'

    elif vote==4:
        pridect_name = 'katarzyna'
    elif vote==5:
        pridect_name = 'konrad'
    elif vote==6:
        pridect_name = 'szczepan'
    else:
        pridect_name='not exist in our system'
    return pridect_name

def testRbf(k1=1.3):
    dataArr, labelArr = load_data_X_Y('samples')
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    filename = 'SVMClassifier_Rbf_' + repr(k1).replace('.', '_') + '.json'

    try:
        SVMClassifier = objectLoadFromFile(filename)
        SVMClassifier.jsonLoadTransfer()  # change back to numpy matrix
        print 'load SVMClassifier successfully'
    except IOError, ValueError:

        print 'SVM classifer file doesnt exist, Train first'
        testSvm = svmTrain(datMat, labelMat, 200, 0.001, ('rbf', k1))
        b, alpha = testSvm.smoP(10000)
        vecAlp, vec, vecClass = testSvm.svmSurpportVecsGet()
        SVMClassifier = svmClassifer(b, vecAlp, vec, vecClass, ('rbf', k1))
        SVMClassifier.jsonDumps(filename)  # change to python default list
        # pdb.set_trace()
        SVMClassifier.jsonLoadTransfer()  # change back to numpy matrix

    m, n = shape(datMat)
    errorCount = 0.0
    for i in range(m):
        result = SVMClassifier.svmClassify(datMat[i, :])
        if result != sign(labelArr[i]):
            print 'training predict result is: ', result, ' training real result is: ', sign(labelArr[i])
            errorCount += 1
    print "the training error rate is: %2.2f%%" % ((float(errorCount) / m) * 100)

    dataArr, labelArr = load_data_X_Y('samples')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        result = SVMClassifier.svmClassify(datMat[i, :])
        if result != sign(labelArr[i]):
            print 'test predict result is: ', result, ' test real result is: ', sign(labelArr[i])
            errorCount += 1
    print "the test error rate is: %2.2f%%" % ((float(errorCount) / m) * 100)

