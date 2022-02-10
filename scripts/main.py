# coding=utf-8
import argparse
import glob
import os
import pickle
import numpy as np

from feature_selection.feature_selection import backwardFeatureSelection,getThreshold
from feature_selection.preprocessing_functions import data_scale
from training import computeAccuracy

#DELTA_GRID = [0,0.0005 , 0.05, 0.25, 0.5, 1.0, 2.0]
DELTA_GRID = [0,0.0005]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int) # k-neighbors
    parser.add_argument("--backward", type=str, default="t")
    parser.add_argument("--nproc", nargs='?', const=1, type=int)
    parser.add_argument("--ScaleAndNoise", type=str, default="ScaleAndNoise")
    parser.add_argument("--classification", nargs='?', const=0, type=int) # setto a 0 così per default considera un problema di regressione 
    parser.add_argument("--filename", type=str) # full path to file, it must be a pickle
    args = parser.parse_args()

    print("NPROC : " + str(args.nproc))
    #input_folder = os.path.join(os.getcwd(), "data", "real")
    #filenames = glob.glob(os.path.join(input_folder, "*.pickle")) # get dataset files in pickle format

    # --- DATA LOAD ---
    # for now this tool only supports a single pickle file
    with open(args.filename, 'rb') as fp:
        dataset = pickle.load(fp)

    # --- FEATURE SELECTION ---
    # X: n rows for m features
    # Y: n rows for l targets
    features = dataset["X"]
    target = dataset["Y"]

    # Scale and Noise if needed
    features = data_scale(features, args.ScaleAndNoise)
    target = data_scale(target, args.ScaleAndNoise)

    # grid with different allowed losses w.r.t. the full features dataset
    grid = DELTA_GRID
    grid.sort()

    # if not specified, fixed fraction of samples
    if args.k is None:
        args.k=len(features)//20

    # set the kind of problem and the dictionary that will store the result
    task = 1 if args.classification == 1 else 0 # task=1 -> classification

    res = {
        "delta" : [], # list with all deltas
        "numSelected" : [], #
        "selectedFeatures" : [] 
    #    "accuracy" : [] # list of scores associated with the reduced problem
    }
    
    for delta in grid:
        print("Current Delta: {0}".format(delta))
        res["delta"].append(delta)
        threshold = getThreshold(task, target, delta) # for the current delta this is the maximum information we want to loose in backward after pruning the worse features
        print("Current Threshold: {0}".format(threshold))
        relevantFeatures = backwardFeatureSelection(threshold,features,target,res, args.k, args.nproc) # CMI feature selection
        res["selectedFeatures"].append(relevantFeatures)
        print("selected Features: {0}".format(res["selectedFeatures"]))
    #    res["accuracy"].append(computeAccuracy(task, relevantFeatures, target)) # performance di un modello lineare 
    
    for i in range(len(res["delta"])):
        print("Delta: {0}, final number of features: {1}, selected features IDs: {2}".format(res["delta"][i], res["numSelected"][i], res["selectedFeatures"][i]))
