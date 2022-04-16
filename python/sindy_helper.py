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
from regression import PolynomialLibrary, TrigLibrary

class Theta:
    def __init__(self,num_snapshots, num_features):
        self.num_snapshots, self.num_features  = num_snapshots , num_features 
        self.feature_names = [f'x{i+1}' for i in range(self.num_features)]
        self.candidate_terms = [ lambda x: torch.ones(self.num_snapshots) ]
        self.candidate_names = ['1']
        self.libs = [ PolynomialLibrary(max_degree=1), TrigLibrary() ]
        for lib in self.libs:
            lib_candidates = lib.get_candidates(self.num_features, self.feature_names)
            for term, name in lib_candidates:
                self.candidate_terms.append(term)
                self.candidate_names.append(name)
    def theta(self,X):
         return torch.stack(tuple(f(X) for f in self.candidate_terms), axis=1)

#         self.SINDy_forward = nn.Linear( len(self.candidate_terms), self.num_features, bias=False)
        
#     def library(self):
#         print('library candidate terms:')
#         return self.candidate_names
    
#     def model_parameters(self):
#         params = list(self.parameters())[0]
#         return params
    
#     def theta(self,X):
#         return torch.stack(tuple(f(X) for f in self.candidate_terms), axis=1)

#     def forward(self):
#         theta_X = self.theta(self.X)
        
#         # X_dot_predict = f(X) = Θ(X)Ξ = Θ(X)[ ξ1, ξ2, ..., ξn ]
#         X_dot_predict = self.SINDy_forward(theta_X)
#         return X_dot_predict