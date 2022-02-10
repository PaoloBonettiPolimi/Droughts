# coding=utf-8
import numpy as np
from multiprocessing import Pool
from feature_selection.mixedRVMI import CMIEstimate,estimateAllMI

def backwardFeatureSelection(threshold,features,target,res,k, nproc):
    'the function returns the selected features starting from the full dataset and removing features keeping the loss of information smaller than the threshold'

    featureScores= []
    relevantFeatures = features # at the beginning all features are included
    idMap = {k: k for k in range(relevantFeatures.shape[1])} # dictionary with original feature position
    CMIScore = 0 # cumulative loss of information
    sortedScores = []

    while CMIScore < threshold and relevantFeatures.shape[1]>1: 
        if nproc > 1:
            featureScores = scoreParallelFeatures(relevantFeatures, target, k, nproc)
        else: 
            featureScores = scoreFeatures(relevantFeatures, target, k) # for each feature it evaluates the I(Y,X_i|X_A), at first step I(Y,X_i|X_{-i}),...
        
        sortedScores = sorted(featureScores, key=lambda x:x[1]) # lista ordinata (ascending) in base al punteggio di ogni feature
        CMIScore += max(sortedScores[0][1],0) # se il punteggio più basso è negativo, prendo 0
        if CMIScore > threshold: break
        relevantFeatures = np.delete(relevantFeatures, sortedScores[0][0], axis=1) # tolgo la feature (column) con punteggio più basso 
        print("Removing original feature: {0}".format(idMap[sortedScores[0][0]])) # original feature position
        for a, b in list(idMap.items())[:-1]: # update of the dictionary storing original positions
            if a >= sortedScores[0][0]:
                idMap[a] = idMap[a+1]
        idMap.pop(max(idMap))
    res["numSelected"].append(relevantFeatures.shape[1]) 
    return list(idMap.values()) 

def forwardFeatureSelection(threshold,features,target,res,k, nproc):
    'returns the relevant features for the target starting from an empty array and populating it with the features that have the best CMI score'

    featureScores=[]
    idMap = {k: k for k in range(features.shape[1])} # dictionary with original feature position
    idSelected = []
    selectedFeatures = [] # empty array at the beginning
    CMIScore = 0 # cumulative loss of information

    firstBest = sorted(estimateAllMI(features, target, k), key=lambda x:x[1], reverse=True) # ordered list (descending) of features MI scores
    selectedFeatures.append(features[:, firstBest[0][0]]) # append the best scoring feature to result

    remainingFeatures = np.delete(features, firstBest[0][0], axis=1) # now score the remaining features

    while CMIScore < threshold and selectedFeatures.shape[1] <= features.shape[1]:
        featureScores = scoreFeatures(remainingFeatures, target, k, np.array(selectedFeatures).T)
        sortedScores = sorted(featureScores, key=lambda x:x[1], reverse=True) # scores in descending order
        CMIScore += max(sortedScores[0][1], 0)
        if CMIScore > threshold : break
        selectedFeatures.append(features[:, sortedScores[0][0]]) # select highest scoring feature
        print("Adding original feature: {0}".format(idMap[sortedScores[0][0]])) # original feature position
        idSelected.append(idMap[sortedScores[0][0]])
        remainingFeatures = np.delete(features, sortedScores[0][0], axis=1) # best scoring no longer needed for evaluation
    res["numSelected"].append(np.array(selectedFeatures).T.shape[1]) 
    return idSelected

def scoreParallelFeatures(features, target, k, nproc, selected=None):
    'Versione con parallelismo dello score'
    args=[]
    for i in range(features.shape[1]):
        if selected is None:
            selected = np.delete(features,i,axis=1)
        args.append((features[:, i], target, selected, k))
    with Pool(nproc) as p:
        scores = p.starmap(CMIEstimate, args)
    scores = np.array(scores)
    return list(zip(range(len(scores)),scores))

def scoreFeatures(features, target, k, selected=None):
    'Ritorna una lista di features ID + punteggio CMI sul dato target'
    scores = np.zeros(features.shape[1])

    for col in range(features.shape[1]):
        if selected is None:
            selected = np.delete(features,col,axis=1)
        scores[col] = CMIEstimate(features[:, col], target, selected, k)
        print("CMI: {0}".format(scores[col]))

    return list(zip(range(len(scores)),scores))

def getThreshold(task, target, delta):
    if task == 1: # classification task
        return (delta**2)/2
    else:
        return delta/2*np.max(target)**2 # l-infinity norm
