from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from util import get_speaker_name
from util import load_data_X_Y
from util import load_data_user_chose
from util import vote_the_max_times
# 76%


def start_GradientBoostingClassifier(wav):
    try:
        clf = joblib.load('model/GB.model')
        print 'load GB model successfully'
    except IOError:

        print 'GB model file doesnt exist, Train first'
        a, b = load_data_X_Y('train')
        # X, y = shuffle_two_list_X_Y(a, b)
        X, y = a, b
        #clf = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,
     max_depth=1, random_state=0)        
        clf.fit(X, y)
        joblib.dump(clf, "model/GB/GB.model")
        print "update the model"

    predict_result = clf.predict(load_data_user_chose(wav))
    vote_vale = vote_the_max_times(predict_result)
    speaker_name = get_speaker_name(vote_vale)
    return speaker_name
test1, test2 = load_data_X_Y('train') 

# print start_calssification_SGD()
def get_the_iter_accucy(max_true_vale=76, model="model/GB/GB.model", iter_times=1):
    try:
        if iter_times > 0:
            print "left " + str(iter_times) + " iteration"
            a, b = load_data_X_Y('train')
            # X, y = shuffle_two_list_X_Y(a, b)
            X, y = a, b
            #clf = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
            clf = GradientBoostingClassifier(n_estimators=10000, learning_rate=1.0,
     max_depth=1, random_state=0)            
            clf.fit(X, y)
            joblib.dump(clf, "model/GB/GB.model")
            print "update the model"

            testX, testY = load_data_X_Y('test')
            true_ans = 0

            for itemX in range(len(testX)):
                predict_result = clf.predict(testX[itemX])
                vote_vale = vote_the_max_times(predict_result)
                if vote_vale == testY[itemX]:
                    true_ans += 1
            true_ans_percent = true_ans * 100 / len(testX)
            print true_ans_percent
            if true_ans_percent > max_true_vale:
                print "good job, the new record:"

                joblib.dump(clf, model)
                print "update the model"
            else:
                true_ans_percent = max_true_vale
            get_the_iter_accucy(true_ans_percent, model, iter_times - 1)
    except ValueError:
        pass
    return 0


def get_accucy(model="model/GB/GB.model"):
    try:
        clf = joblib.load(model)
        print 'load GB model successfully'
    except IOError:

        print 'GB model file doesnt exist, Train first'
        a, b = load_data_X_Y('train')
        # X, y = shuffle_two_list_X_Y(a, b)
        X, y = a, b        
        #clf = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,
     max_depth=1, random_state=0)        
        clf.fit(X, y)
        joblib.dump(clf, "model/GB/GB.model")
        print "update the model"
    testX, testY = load_data_X_Y('test')
    true_ans = 0

    for itemX in range(len(testX)):
        predict_result = clf.predict(testX[itemX])
        vote_vale = vote_the_max_times(predict_result)
        if vote_vale == testY[itemX]:
            true_ans += 1
    true_ans_percent = true_ans * 100 / len(testX)

    return true_ans_percent

# get_the_iter_accucy()
print get_accucy()