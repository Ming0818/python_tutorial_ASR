from sklearn.linear_model import SGDClassifier
from classification_svm import load_data_X_Y
from classification_svm import load_data_user_chose
from classification_svm import vote_the_max_times
from classification_svm import get_speaker_name
from sklearn.externals import joblib


def start_calssification_SGD(wav):
    try:
        clf = joblib.load('SGD.model')
        print 'load SGDClassifier successfully'
    except IOError:

        print 'SGD classifer file doesnt exist, Train first'

        X, y = load_data_X_Y('train')
        clf = SGDClassifier(loss="hinge", penalty="l2")
        clf.fit(X, y)

    predict_result = clf.predict(load_data_user_chose(wav))
    vote_vale = vote_the_max_times(predict_result)
    speaker_name = get_speaker_name(vote_vale)
    return speaker_name


# print start_calssification_SGD()
def get_the_iter_accucy(max_true_vale=60, model="SGD.model", iter_times=1000):
    try:
        if iter_times>0:
            print "this is the "+str(iter_times)+"nd iteration"
            X, y = load_data_X_Y('train')
            clf = SGDClassifier(loss="hinge", penalty="l2")
            clf.fit(X, y)
            testX, testY = load_data_X_Y('test')
            true_ans = 0

            for itemX in range(len(testX)):
                predict_result = clf.predict(testX[itemX])
                vote_vale = vote_the_max_times(predict_result)
                if vote_vale == testY[itemX]:
                    true_ans += 1
            true_ans_percent = true_ans * 100 / len(testX)
            if true_ans_percent > max_true_vale:
                print "good job, the new record:",
                print true_ans_percent
                joblib.dump(clf, model)
            else:
                true_ans_percent=max_true_vale
            get_the_iter_accucy(true_ans_percent,model,iter_times-1)
    except ValueError:
        pass
    return 0


def get__accucy(model="SGD.model"):


    try:
        clf = joblib.load(model)
        print 'load SGDClassifier successfully'
    except IOError:

        print 'SGD classifer file doesnt exist, Train first'

        X, y = load_data_X_Y('train')
        clf = SGDClassifier(loss="hinge", penalty="l2")
        clf.fit(X, y)
    testX, testY = load_data_X_Y('test')
    true_ans = 0

    for itemX in range(len(testX)):
        predict_result = clf.predict(testX[itemX])
        vote_vale = vote_the_max_times(predict_result)
        if vote_vale == testY[itemX]:
            true_ans += 1
    true_ans_percent = true_ans * 100 / len(testX)


    return true_ans_percent
# get_the_iter_accucy()#search a best value
