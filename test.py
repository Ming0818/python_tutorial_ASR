import scipy
from numpy import zeros
from sklearn import svm
import scipy.io.wavfile as wf
from mfcc_feature import mfcc
# X = [[1],[4],[3],[5],[2]]
# Y = [1, 16,9,25,4]


sr,data=wf.read('english.wav')
feat_mfcc=mfcc(data,sr)
X=feat_mfcc
Y= scipy.ones(len(feat_mfcc))
Y[1]=0
print Y
clf = svm.SVC()
clf.fit(X, Y)
print feat_mfcc[1:3,:]
print clf.get_params()
print clf.score([[2,1,2,3,2,2,3,2,9,1,2,3,4],[2,1,2,3,2,2,3,4,2,1,2,3,4]],[1,1])
print clf.predict([2,1,2,3,2,2,3,2,9,1,2,3,4])
print clf.predict([2,1,2,3,2,2,3,4,2,1,2,3,4])