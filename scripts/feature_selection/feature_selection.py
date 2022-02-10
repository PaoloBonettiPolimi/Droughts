# coding=utf-8
import numpy as np
from multiprocessing import Pool
from feature_selection.mixedRVMI import CMIEstimate

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

def scoreParallelFeatures(features, target, k, nproc):
    'Versione con parallelismo dello score'
    args=[]
    for i in range(features.shape[1]):
        args.append((features[:, i], target, np.delete(features,i,axis=1), k))
    with Pool(nproc) as p:
        scores = p.starmap(CMIEstimate, args)
    scores = np.array(scores)
    return list(zip(range(len(scores)),scores))

def scoreFeatures(features, target, k):
    'Ritorna una lista di features ID + punteggio CMI sul dato target'
    scores = np.zeros(features.shape[1])

    for col in range(features.shape[1]):
        scores[col] = CMIEstimate(features[:, col], target, np.delete(features,col,axis=1), k)
        print("CMI: {0}".format(scores[col]))

    return list(zip(range(len(scores)),scores))

def getThreshold(task, target, delta):
    if task == 1: # classification task
        return (delta**2)/2
    else:
        return delta/2*np.max(target)**2 # l-infinity norm
