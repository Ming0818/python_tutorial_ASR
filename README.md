# Auto Speaker Recognition

# main.py

the main file for test

# audio_record.py

record audio from micro phone

# count_days.py

count days between two date. 20110805   20160903

# mfcc_feature.py

extract mfcc feature from wav files

# SGD.model*

the trained model on train set , and the accurate is 70%

# util.py

contains the most useful functions 

# train

train data is 75% of all the data

# test 

test data is 25% of all the data and has no overlap with train set

# classification_SGD.py

is the main classification function py file , and it used the sklearn's SGD    
niter was set 10000 could get 70% of accurate.
# classification_BNB.py
this is the sklern naive_bayes BernoulliNB ,
and it reach to just 56%
# classification_DT.py
this is the sklern tree.DecisionTreeClassifier ,
and it reach to just 63%
# classification_GB.py
this is the sklern GradientBoostingClassifier,
and it reach to the best now of 76% when n_estimators=1000,
but it produce too many model components to store.
# classification_GNB.py
this is the sklern naive_bayes GaussianNB,
and it reach to just 63%
# vote_result.py
add a vote decsion , every method have the acurrcy number ticiks to vote the final answer.
after the vote , we achived  96% at test set.


# [beta1.0](https://github.com/zhangxulong/python_tutorial_ASR/releases)
 



