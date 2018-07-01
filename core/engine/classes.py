import numpy as np


class Cluster:
    def __init__(self, points=np.array([[], []]), cores=np.array([[], []]), mean=None):
        self.points = points
        self.cores = cores
        self.mean = mean


class Noise:
    def __init__(self, points=np.array([])):
        self.points = points
