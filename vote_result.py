import os

from classification_BNB import start_calssification_BNB
# 56%
from classification_DT import start_calssification_DT
# 63%
from classification_GB import start_GradientBoostingClassifier
# 76%
from classification_GNB import start_calssification_GNB
# 63%
from classification_SGD import start_calssification_SGD


# 70%

def check_the_ASR_acurrcy():
    dir_path_to_dataset = 'test'
    true_ans = 0
    total = 0

    for parent, dirnames, filenames in os.walk(dir_path_to_dataset):
        for filename in filenames:
            if '.wav' in filename:
                total = total + 1
                wav_file_path = os.path.join(parent, filename)

                BNB_predict_result = start_calssification_BNB(wav_file_path)
                DT_predict_result = start_calssification_DT(wav_file_path)
                GB_predict_result = start_GradientBoostingClassifier(wav_file_path)
                GNB_predict_result = start_calssification_GNB(wav_file_path)
                SGD_predict_result = start_calssification_SGD(wav_file_path)

                vote_result_of_all = [BNB_predict_result] * 56 + [DT_predict_result] * 63 + [GB_predict_result] * 76 + [
                                                                                                                           GNB_predict_result] * 63 + [
                                                                                                                                                          SGD_predict_result] * 70
                items = dict([(vote_result_of_all.count(i), i) for i in vote_result_of_all])
                max_vale = items[max(items.keys())]

                if max_vale in wav_file_path:
                    true_ans += 1
    true_ans_percent = true_ans * 100 / total

    print true_ans_percent
    # 96%
    return true_ans_percent


def use_the_ASR(wav_file_path):
    BNB_predict_result = start_calssification_BNB(wav_file_path)
    DT_predict_result = start_calssification_DT(wav_file_path)
    GB_predict_result = start_GradientBoostingClassifier(wav_file_path)
    GNB_predict_result = start_calssification_GNB(wav_file_path)
    SGD_predict_result = start_calssification_SGD(wav_file_path)

    vote_result_of_all = [BNB_predict_result] * 56 + \
                         [DT_predict_result] * 63 + \
                         [GB_predict_result] * 76 + \
                         [GNB_predict_result] * 63 + \
                         [SGD_predict_result] * 70
    items = dict([(vote_result_of_all.count(i), i) for i in vote_result_of_all])
    max_vale = items[max(items.keys())]

    return max_vale
