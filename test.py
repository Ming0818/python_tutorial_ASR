import os

from sklearn import svm
import scipy.io.wavfile as wf
from mfcc_feature import mfcc
from sklearn.externals import joblib


# X = [[1],[4],[3],[5],[2]]
# Y = [1, 16,9,25,4]

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

def classification_SVC(
        predictX=[12.540124391170103, -15.463048253688235, 22.303617297458864, 14.00664884709655, 17.47925833621953,
                  7.701722714394731, 5.171859887375148, -3.463212292168566, -0.4483612715659834, 3.3745620890437134,
                  6.483955957515144, -2.174288845618787, -2.679758262971649]
        , model=""):
    if model != "":
        clf = joblib.load(model)
    else:
        print "else"
        # clf = svm.SVC()
        clf=svm.SVR()
        inputX, labeY = load_data_X_Y('samples')
        clf.fit(inputX, labeY)
        joblib.dump(clf, "train_model.m")

    predictY = clf.predict(predictX)
    print clf.get_params()
    return predictY
# print classification_SVC(load_data_user_chose('01.wav'),"train_model.m")
# print classification_SVC(load_data_user_chose('02.wav'),"train_model.m")
# print classification_SVC(load_data_user_chose('03.wav'),"train_model.m")
print classification_SVC(load_data_user_chose('samples/adam_traba/imie/06.wav'),"train_model.m")
# print classification_SVC(load_data_user_chose('01.wav'),"train_model.m")