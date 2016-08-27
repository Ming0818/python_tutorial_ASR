import json


def object2dict(obj):
    #convert object to a dict   
    d = {'__class__':obj.__class__.__name__, '__module__':obj.__module__}   
    d.update(obj.__dict__)   
    return d

def objectDumps2File(obj, jsonfile):
    objDict = object2dict(obj)
    with open(jsonfile, 'w') as f:
        f.write(json.dumps(objDict))
    
def dict2object(d):   
    '''convert dict to object, the dict will be changed'''    
    if'__class__' in d:   
        class_name = d.pop('__class__')   
        module_name = d.pop('__module__')   
        module = __import__(module_name)   
        #print 'the module is:', module   
        class_ = getattr(module,class_name)   
        args = dict((key.encode('ascii'), value) for key, value in d.items()) #get args  
        #print 'the atrribute:', repr(args)
        #pdb.set_trace()
        inst = class_(**args) #create new instance   
    else:   
        inst = d   
    return inst

def objectLoadFromFile(jsonFile):
    '''load json file and generate a new object instance whose __name__ filed
    will be 'inst' '''
    with open(jsonFile) as f:
        objectDict =json.load(f)
    obj = dict2object(objectDict)
    return obj


