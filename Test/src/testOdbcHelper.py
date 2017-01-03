'''
Created on Sep 25, 2016

@author: Jacinto
'''

from src.odbcHelper import buildConnectionString



class MyClass(object):
    '''
    classdocs
    '''
    '''odbcHelper.buildConnectionString({"server": "myServer", "database": "master", "uid": "jac", "pwd": "secreto"})'''
    paramsTest = {"server": "mpilgrim", "database": "master", "uid": "sa", "pwd": "secret"}
    buildConnectionString({"server": "myServer", "database": "master", "uid": "jac", "pwd": "secreto"})
    '''odbcHelper.__name__'''

    def __init__(self, params):
        '''
        Constructor
        '''
