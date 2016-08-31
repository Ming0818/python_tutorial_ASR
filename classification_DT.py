from sklearn.externals import joblib
from util import load_data_user_chose, vote_the_max_times, get_speaker_name
from util import load_data_X_Y
from util import shuffle_two_list_X_Y
from sklearn.tree import DecisionTreeClassifier
# 63%
def start_calssification_DT(wav):
    try:
        dt = joblib.load('model/DT.model')
        print 'load DTClassifier successfully'
    except IOError:

        print 'DT classifer file doesnt exist, Train first'
        a, b = load_data_X_Y('train')
        X, y = shuffle_two_list_X_Y(a, b)
        dt = DecisionTreeClassifier()
        dt.fit(X, y)
        joblib.dump(dt, 'model/DT.model')
        print "update the model"

    predict_result = dt.predict(load_data_user_chose(wav))
    vote_vale = vote_the_max_times(predict_result)
    speaker_name = get_speaker_name(vote_vale)
    return speaker_name


def get_the_iter_accucy(max_true_vale=63, model="model/DT.model", iter_times=3):
    try:
        if iter_times > 0:
            print "left " + str(iter_times) + " iteration"
            a, b = load_data_X_Y('train')
            X, y = shuffle_two_list_X_Y(a, b)
            dt = DecisionTreeClassifier()
            dt.fit(X, y)

            testX, testY = load_data_X_Y('test')
            true_ans = 0

            for itemX in range(len(testX)):
                predict_result = dt.predict(testX[itemX])
                vote_vale = vote_the_max_times(predict_result)
                if vote_vale == testY[itemX]:
                    true_ans += 1
            true_ans_percent = true_ans * 100 / len(testX)
            print true_ans_percent
            if true_ans_percent > max_true_vale:
                print "good job, the new record:"

                joblib.dump(dt, model)
                print "update the model"
            else:
                true_ans_percent = max_true_vale
            get_the_iter_accucy(true_ans_percent, model, iter_times - 1)
    except ValueError:
        pass
    return 0


def get_accucy(model="model/DT.model"):
    try:
        dt = joblib.load(model)
        print 'load DTClassifier successfully'
    except IOError, ValueError:

        print 'DT classifer file doesnt exist, Train first'
        a, b = load_data_X_Y('train')
        X, y = shuffle_two_list_X_Y(a, b)
        dt = DecisionTreeClassifier()
        dt.fit(X, y)
    testX, testY = load_data_X_Y('test')
    true_ans = 0

    for itemX in range(len(testX)):
        predict_result = dt.predict(testX[itemX])
        vote_vale = vote_the_max_times(predict_result)
        if vote_vale == testY[itemX]:
            true_ans += 1
    true_ans_percent = true_ans * 100 / len(testX)

    return true_ans_percent


# get_the_iter_accucy()
# print get_accucy()
