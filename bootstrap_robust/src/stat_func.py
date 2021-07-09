import numpy as np
import scipy.stats


def mean(values, axis=0):
    '''Returns the mean for each row of a matrix'''
    return np.mean(values, axis=axis)

def median(values, axis=0):
    '''Returns the median for each row of a matrix'''
    return np.median(values, axis=axis)


def trim_mean(values, proportiontocut, axis=0):
    '''Returns the trimmed mean for each row of a matrix'''
    return scipy.stats.trim_mean(values, proportiontocut, axis=axis)
