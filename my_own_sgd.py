# Add Spark Python Files to Python Path
import sys
import os
import random
import math

SPARK_HOME = "/usr/local/spark-0.9.1" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path


from pyspark.mllib.classification import LogisticRegressionWithSGD
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
import pyspark

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = pyspark.SparkContext() 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    # labels must be at the beginning for LRSGD
    label = feats[len(feats) - 1]
    
    feats = feats[: len(feats) - 1]
    #add x0=1
    feats.insert(0,1)
    #
    #feats.insert(0,label)
    features = [ float(feature) for feature in feats ] # need floats
    
    return LabeledPoint(label, features)

def theta(s):
    a=1/(1+math.exp(-s))
    return a

def SGD_log(parsedData,ita,trainSize,iteration_time,dimension):
    w=[0]*(dimension+1)
    for i in range(iteration_time):
        #choose=random.randint(0,trainSize-1)
        temp=parsedData.takeSample(False, 1)
        #print("features=",temp[0].features)
        #print("label=",temp[0].label)
        if temp[0].label==0:
            temp[0].label=-1
        #print("label=",temp[0].label)
        w=w+ita*theta(-temp[0].label*np.dot(w,temp[0].features))*temp[0].label*temp[0].features
        print("i= ",i," w= ",w)
    #ans=eca(y,w,x,trainSize)
    #print(numData.reduce(lambda x,y:x+y)/numData.count())
    ans=parsedData.filter( lambda x: np.dot(x.features,w)*(x.label-0.5)<=0 ).count()/float(parsedData.count())
    
    
    return ans

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("gs://vm_hw4/data_banknote_authentication.txt")
parsedData = data.map(mapper)
dimension=4
ita=1
#test
iteration_time=100
trainSize=parsedData.count()
print("Training Error = " + str(SGD_log(parsedData,ita,trainSize,iteration_time,dimension)))

"""
# Train model
model = LogisticRegressionWithSGD.train(parsedData)

# Predict the first elem will be actual data and the second 
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: (int(point.label), 
        model.predict(point.features)))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))
"""