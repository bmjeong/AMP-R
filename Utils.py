import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import time
import copy
import math
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import lkh

class Depot:
    def __init__(self, id, Vehicles, position):
        self.id = id
        self.vehicle_n = len(Vehicles)
        self.Vehicles = Vehicles
        for i, v in enumerate(self.Vehicles):
            v.id = i
        self.position = position ## np.array 여야 함

class Vehicle:
    def __init__(self, id, type_, vel, cap):
        self.id = id
        self.type_ = type_
        self.vel = vel
        self.cap = cap

class Customer:
    def __init__(self, id, demand, position):
        self.id = id
        self.demand = demand
        self.position = position ## np.array 여야 함