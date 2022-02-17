# coding=utf-8
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.utils import check_random_state

def data_scale(features, flag):
    if features.ndim == 1:
        features = features.reshape(-1,1)
    if flag=='ScaleAndNoise':
        x = preprocessing.scale(features, with_mean=False, copy=False)        
        rng = check_random_state(None)
        means = np.maximum(1, np.mean(np.abs(features), axis=0))
        x += (1e-10 * means * rng.randn(x.shape[0], x.shape[1]))        
        return x
    if flag=='Scale':
        x = preprocessing.scale(features, with_mean=False, copy=False)        
        return x
    return features
