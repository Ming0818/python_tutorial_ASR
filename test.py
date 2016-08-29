import os

for parent, dirnames, filenames in os.walk("/Users/zhangxulong/"
                                           "Desktop/pythonTutorial/"
                                           "python_tutorial"
                                           "_AS"
                                           "R"):

    for filename in filenames:
        path = os.path.join(parent, filename)
        if 'GBC.model_' in filename:

            os.remove(path)
print "ok"