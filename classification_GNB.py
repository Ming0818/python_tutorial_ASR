from sklearn.externals import joblib
from util import load_data_user_chose, vote_the_max_times, get_speaker_name
from util import load_data_X_Y
from util import shuffle_two_list_X_Y
from sklearn.naive_bayes import GaussianNB
# 63%   76 70 63 63 56
def start_calssification_GNB(wav):
    try:
        gnb = joblib.load('model/GNB.model')
        print 'load GNBClassifier successfully'
    except IOError:

        print 'GNB classifer file doesnt exist, Train first'
        a, b = load_data_X_Y('train')
        X, y = shuffle_two_list_X_Y(a, b)
        gnb = GaussianNB()
        gnb.fit(X, y)
        joblib.dump(gnb, 'model/GNB.model')
        print "update the model"

    predict_result = gnb.predict(load_data_user_chose(wav))
    vote_vale = vote_the_max_times(predict_result)
    speaker_name = get_speaker_name(vote_vale)
    return speaker_name


def get_the_iter_accucy(max_true_vale=70, model="model/GNB.model", iter_times=3):
    try:
        if iter_times > 0:
            print "left " + str(iter_times) + " iteration"
            a, b = load_data_X_Y('train')
            X, y = shuffle_two_list_X_Y(a, b)
            gnb = GaussianNB()
            gnb.fit(X, y)

            testX, testY = load_data_X_Y('test')
            true_ans = 0

            for itemX in range(len(testX)):
                predict_result = gnb.predict(testX[itemX])
                vote_vale = vote_the_max_times(predict_result)
                if vote_vale == testY[itemX]:
                    true_ans += 1
            true_ans_percent = true_ans * 100 / len(testX)
            print true_ans_percent
            if true_ans_percent > max_true_vale:
                print "good job, the new record:"

                joblib.dump(gnb, model)
                print "update the model"
            else:
                true_ans_percent = max_true_vale
            get_the_iter_accucy(true_ans_percent, model, iter_times - 1)
    except ValueError:
        pass
    return 0


def get_accucy(model="model/GNB.model"):
    try:
        gnb = joblib.load(model)
        print 'load GNBClassifier successfully'
    except IOError, ValueError:

        print 'GNB classifer file doesnt exist, Train first'
        a, b = load_data_X_Y('train')
        X, y = shuffle_two_list_X_Y(a, b)
        gnb = GaussianNB()
        gnb.fit(X, y)
    testX, testY = load_data_X_Y('test')
    true_ans = 0

    for itemX in range(len(testX)):
        predict_result = gnb.predict(testX[itemX])
        vote_vale = vote_the_max_times(predict_result)
        if vote_vale == testY[itemX]:
            true_ans += 1
    true_ans_percent = true_ans * 100 / len(testX)

    return true_ans_percent


# get_the_iter_accucy()
# print get_accucy()
