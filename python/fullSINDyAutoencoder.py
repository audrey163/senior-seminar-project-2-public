import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pprint as pp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

from regression import PolynomialLibrary, TrigLibrary
import sindy_helper
from dynamicalsystems import TrainDataset

class FullSINDyAutoencoder(nn.Module):
    def __init__(self,batch_size, num_features):
        super().__init__()
        self.num_snapshots, self.num_features = batch_size, 2
        self.Theta = sindy_helper.Theta(self.num_snapshots, self.num_features)
        self.theta = self.Theta.theta
        self.encoder = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, self.num_features) # -> N, 3
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.num_features,4),
            nn.ReLU(),
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,10)
        )
        self.SINDy_layer = nn.Sequential(
            nn.Linear(len(self.Theta.candidate_terms),self.num_features,bias=False)
        )
    def forward(self, x, dx):
        Z = self.encoder(x)
        dZ = self.get_dZ(x,dx)
        X_pred = self.decoder(Z)
        dZ_pred = self.sindy(Z)
        #dX_pred = self.get_dX(Z,dZ)
        Xi = self.SINDy_layer[0].weight
        #This is what I need I dont know how to get Xi = Weghts from self.SINDy_layer = nn.Linear..
        return { 'X' : x, 'dX' : dx,'Z' : Z, 'dZ' : dZ,'X_pred' : X_pred, 'dZ_pred' : dZ_pred, 'Xi' : Xi} #'dX_pred' : dX_pred
    def get_dZ(self,x,dx):
        J = torch.autograd.functional.jacobian(self.encoder, x)
        return torch.matmul(J.T,dx)
 
    def get_dX(self,z,dz):
        J = torch.autograd.functional.jacobian(self.decoder, z)
        return torch.matmul(J.T,dz)

    def sindy(self,Z): #HW 1
         theta_Z = self.theta(Z)
         # Z_dot_predict = f(Z) = Θ(Z)Ξ = Θ(Z)[ ξ1, ξ2, ..., ξn ]
         return self.SINDy_layer(theta_Z)

    def loss(self,args,reg):
        return reg['X']*torch.linalg.norm(args['X'] - args['X_pred']) 
        + reg['SINDy']*torch.linalg.norm(args['dZ_pred'] - args['dZ'])
        + reg['dX']*torch.linalg.norm(args['dX_pred'] - args['dX'])
        + reg['Xi']*torch.linalg.norm(args['Xi'],ord=1)