import torch
from torchdiffeq import odeint
import geoopt.manifolds.stereographic.math as gmath
from PIL import Image
import matplotlib.pyplot as plt
import os
from argparse import Namespace
from manifm.manifolds.hyperbolic import PoincareBall as PB




# ----------------------------------------------------------
# Load FM model
# ----------------------------------------------------------
from manifm.eval_utils import load_model
checkpoint = "/home/fvaleau/riemannian-fm/outputs/runs/images/fm/2025.12.15/120733/checkpoints/last.ckpt"
cfg, fm_model = load_model(checkpoint)



fm_model = fm_model.cuda().eval()
fm_model.manifold = PB().cuda()
dim = fm_model.dim


# ----------------------------------------------------------
# Sample from wrapped normal
# ----------------------------------------------------------
def sample_base(n, std):
    #return fm_model.manifold.random_base(n, dim, std).cuda()
    return PB.random_base(n, dim, std).cuda()


# ----------------------------------------------------------
# MAIN: sample 8 points & visualize flow
# ----------------------------------------------------------

n_samples = 8
z0s = sample_base(n_samples, std=0.03)
trajs = []
indexes =[0, 72, 334, 500,606, 781, 888, 949, 981, 995, 1000]

#for z0 in z0s:
for i, z0 in enumerate(z0s):
    traj = fm_model.sample_all(1, device="cuda", x0=z0.unsqueeze(0))   # [1001, 1, dim]
    traj_subset = traj[indexes, 0, :]      #only 11 samples along the trajectory        
    traj_subset = traj_subset.squeeze(1)  
    trajs.append(traj_subset)

trajs = torch.stack(trajs).cuda()
print("trajs.shape: ", trajs.shape)  
torch.save(trajs, "trajs")



