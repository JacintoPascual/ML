'''
Created on Sep 24, 2016

@author: Jacinto
'''
def buildConnectionString(params):
    """Build a connection string from a dictionary of parameter 
    
Returns string."""
#Build a connection string from a dictionary of parameter Returns string."""
    return ";".join(["%s=%s" % (k, v) for k, v in params.items()])
    
if __name__ == "__main__":
    myParams = {"server":"mpilgrim", \
"database":"master", \
"uid":"sa", \
"pwd":"secret" \
}
    myParams["pwd"] = "secretisimo"
    # Del one elment
    #del myParams["pwd"]
    # Remove everything
    #myParams.clear()
print (buildConnectionString(myParams))         