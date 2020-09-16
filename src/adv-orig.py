import torch
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F

import hashlib
def md5sum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


## Download Z+jets dataset from Zenodo, or use a local cache stored at zjets_sig.npy and zjets_bg.npy
class ZJetsDataset(torch.utils.data.Dataset):
    def __init__(self, nevt=None, pt_min=5e-4, ntrk_min=10, ntrk_max=64, use_cache=False, device=None):
        if use_cache:
            try:
                self.sig = np.load("zjets_sig.npy")
                self.bg = np.load("zjets_bg.npy")
            except FileNotFoundError:
                raise FileNotFoundError("Cache files zjets_sig.npy and/or zjets_bg.npy are missing! Please run with use_cache=False, and then generate a cache with ZJetsDataset.cache()")
        else:
            sig, bg = self._get_files()

            sig = sig['constituents'][:460000]
            bg = bg['constituents'][:460000]

            sig[sig[:,:,0]<pt_min] = 0
            bg[bg[:,:,0]<pt_min] = 0
            
            ntrk_sig = np.sum(sig[:,:,0]>0,axis=-1)
            ntrk_bg = np.sum(bg[:,:,0]>0, axis=-1)

            self.sig = sig[ntrk_sig>=ntrk_min][:400000,:ntrk_max]
            self.bg = bg[ntrk_bg>=ntrk_min][:400000,:ntrk_max]
        
        if nevt is not None:
            self.sig = self.sig[:nevt//2]
            self.bg = self.bg[:nevt//2]
        
        x = np.concatenate([self.bg, self.sig], axis=0)
        y = np.concatenate([np.zeros(len(self.bg)), np.ones(len(self.sig))])
                
        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.int64, device=device)
    
    def cache(self):
        np.save("zjets_sig.npy", self.sig)
        np.save("zjets_bg.npy", self.bg)
        
    def _get_files(self):
        files = [
            {
                'file': 'zjets_full_yz.npz',
                'url': 'https://zenodo.org/record/3627324/files/particles_yz.npz?download=1',
                'md5': '7939a1fc817e16a72bfd60cf203bb31d'
            },
            {
                'file': 'zjets_full_jj.npz',
                'url': 'https://zenodo.org/record/3627324/files/particles_jj.npz?download=1',
                'md5': '6a3af195c2998c707a29af1647ae6dfc'
            }
        ]
        
        d = []
        for f in files:
            if not os.path.exists(f['file']):
                print("Downloading %s from %s"%(f['file'], f['url']))
                torch.hub.download_url_to_file(f['url'], f['file'])
            if not md5sum(f['file']) == f['md5']:
                raise ValueError("Failed md5 checksum for file: %s"%f['file'])
            d.append(np.load(f['file']))
        return d
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        
        return (self.x[idx], self.y[idx])
    
def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

# convert (pt,eta,phi) to (E, px, py, pz)
# input shape: (d1, d2, ... , dn, 3)
# output shape: (d1, d2, ... , dn, 4)
def to_rect(x):
    #x = x.double()
    pt, eta, phi = torch.split(x, 1, dim=-1)[:3]

    px = pt*torch.cos(phi)
    py = pt*torch.sin(phi)
    pz = pt*torch.sinh(eta)
    e = pt*torch.cosh(eta)

    p4 = torch.cat([e, px, py, pz], axis=-1)
    p4[pt.squeeze(axis=-1)==0] = 0
    
    return p4#.float()

# convert (E, px, py, pz) to (pt, eta, phi, mass)
# input shape: (d1, d2, ... , dn, 4)
# output shape: (d1, d2, ... , dn, 4)
def to_det(x):
    #x = x.double()
    e, px, py, pz = torch.split(x, 1, dim=-1)
    
    pt2 = px**2 + py**2
    pt = torch.sqrt(pt2)
    p2 = pt2 + pz**2
    
    phi = torch.atan2(py, px)
    
    eta = atanh(pz / torch.sqrt(p2))
    
    m = torch.sqrt(e**2 - p2)
    
    #return m.float() #torch.sqrt(m2).float()
    p4 = torch.cat([pt, eta, phi, m], axis=-1)
    p4[e.squeeze(axis=-1)==0] = 0
    
    return p4

# convert a list of [(pt,eta,phi),...] to a jet (pt,eta,phi,M).
# input shape: (d1, d2, ... , dn-1, dn, 3)
# output shape: (d1, d2, ..., dn-1, 4)
def to_jet(x):
    p4 = to_rect(x)
    
    jp4 = p4.sum(axis=-2)
    jp4 = to_det(jp4)
    return jp4

# center the list of particles x about the axis of jet j
# input shape:
#              x - (d1, d2, ... , dn-1, dn, 3)
#              j - (d1, d2, ... , dn-1, 4)
# output shape: (d1, d2, ..., dn-1, dn, 3)
def center_on(j, x):
    jpt, jeta, jphi, jm = torch.split(j, 1, dim=-1)
    pt0, eta0, phi0 = torch.split(x, 1, dim=-1)
    
    # we can simpliy shift the particles' eta values
    eta = eta0 - jeta.unsqueeze(-2)
    
    # for phi values, to avoid issues of 2pi-wrapping
    #phi = torch.fmod(phi0 - jphi.unsqueeze(-2) + np.pi, 2*np.pi) - np.pi
    #phi = phi0 - jphi.unsqueeze(-2)
    #phi = torch.fmod(phi0 - jphi.unsqueeze(-2), np.pi)
    
    jphi = jphi.unsqueeze(-1)
    px0 = pt0*torch.cos(phi0)
    py0 = pt0*torch.sin(phi0)
    
    px = px0*torch.cos(jphi) + py0*torch.sin(jphi)
    py = py0*torch.cos(jphi) - px0*torch.sin(jphi)
    phi = torch.atan2(py, px)
    
    phi[pt0==0] = 0
    eta[pt0==0] = 0
    
    # pt is invariant to rotations in both eta and phi (NB: energy is not!)
    pt = pt0
    
    return torch.cat([pt, eta, phi], axis=-1)

# convenience method to center particles x about the jet defined by themselves.
# needs to calculate the jet kinematics each time, though, so inefficient if
# you've already calculated or are using the corresponding jet.
# input shape: (d1, d2, ... , dn-1, dn, 3)
# output shape: (d1, d2, ... , dn-1, dn, 3)
def center(x):
    j = to_jet(x)
    return center_on(j, x)


# Simple deep-sets type classifier.
# It expects as input a tensor with shape (d1, d2, ... , dn-1, dn, 3).
# In our case "dn" should correspond to the number of particles, which will get summed over.
# The latent space encoding will have shape (d1, d2, ... , dn-1, L),
# where L is a hyperparameter representing the dimensionality of the latent space.
# The output will have shape (d1, d2, ..., dn-1, 2), where the last dimension are
# raw logits for binary crossentropy.
class JetClassifier(torch.nn.Module):
    def __init__(self, L=32, u=128):
        super(JetClassifier, self).__init__()
        
        # "inner" network. An identical copy of
        # this subnetwork is applied to each input
        # independently.
        self.f1 = torch.nn.ModuleList([
            nn.Linear(3,u),
            nn.Dropout(0.25),
            nn.Linear(u, u),
            nn.Dropout(0.25),
            nn.Linear(u, L),
        ])
        
        # "outer" network. Simple dense network that
        # acts on the L-dimensional summed latent space
        # encoding, and outputs 2 categorical logits for
        # classification score.
        self.f2 = torch.nn.ModuleList([
            nn.Linear(L, u),
            nn.Linear(u, u),
            nn.Linear(u, 2),
        ])
        
    def forward(self, x0):
        # we peel off the pT of each input particle, in order
        # to define a mask. Only particles with nonzero pT
        # will contribute to the latent space sum.
        # this is a convenient way to handle a variable-length input
        # with a fixed input tensor size.
        pt0 = torch.split(x0, 1, dim=-1)[0]
        
        # compute the jet kinematics for the particles in x0
        j = to_jet(x0)
        # center the particles about their jet axis
        x = center_on(j, x0)
        
        for l in self.f1[:-1]:
            x = F.relu(l(x))
        x = self.f1[-1](x)
        
        x = torch.sum((pt0>0)*x, axis=-2)
        
        for l in self.f2[:-1]:
            x = F.relu(l(x))
        x = self.f2[-1](x)
        return x
    
    
    
# function to make fancy animations of jets
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def animate_jet(jets, preds, ytrue):
    fig = plt.figure(figsize=plt.figaspect(1))
    
    j = jets[0]
    j = j[j[:,0]>0]

    c1 = plt.Circle((0,0), 1, color='aliceblue', fill=True)
    c2 = plt.Circle((0,0), 1, ls='--', lw=2, color='gray', fill=False)
    plt.gca().add_artist(c1)
    plt.gca().add_artist(c2)
    
    sc = plt.scatter(j[:,1], j[:,2], s=(j[:,0]*3e3)**1., cmap='magma', c=np.log(j[:,0]), vmin=np.log(5e-5), vmax=np.log(0.5), zorder=2)

    
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.gca().axis('off')
    plt.title(r"$y_{true}=%d\ \ \ \ y_{pred}=%.3f$\ \ \ \ y_{adv}=%.3f" % (ytrue, preds[0], preds[0]))
    
    def update(i):
        j = jets[i]
        j = j[j[:,0]>0]
    
        sc.set_offsets(j[:,1:3])
        sc.set_sizes(j[:,0]*3e3)
        plt.title(r"$y_{true}=%d\ \ \ \ y_{pred}=%.3f\ \ \ \ y_{adv}=%.3f$" % (ytrue, preds[0], preds[i]))
        return sc,
    
    ani = animation.FuncAnimation(fig, update, frames=len(jets)).to_html5_video()
    plt.close()
    return ani


def pgd_attack(model, x, y, n_iter, n_trials, eps, alpha, return_max=True, return_history=False, random_init=True):
    #print('x     ', x.shape)
    
    xadv = x.detach().unsqueeze(1)
    #print('xadv  ', xadv.shape)
    
    if random_init:
        dist = torch.distributions.Uniform(torch.tensor(-1.).to(x.device),torch.tensor(1.).to(x.device))
        deltas = dist.sample((x.shape[0], n_trials) + x.shape[1:])
    else:
        deltas = torch.zeros((x.shape[0], n_trials) + x.shape[1:], device=x.device)
    
    xpt, xeta, xphi = torch.chunk(xadv, 3, axis=-1)
    dpt, deta, dphi = torch.chunk(deltas, 3, axis=-1)
    
    mask = xpt>0
    
    x_lo = xadv.clone()
    x_hi = xadv.clone()
    xpt_lo, xeta_lo, xphi_lo = torch.chunk(x_lo, 3, axis=-1)
    xpt_hi, xeta_hi, xphi_hi = torch.chunk(x_hi, 3, axis=-1)
    
    xpt_lo[:] *= (1-eps[0])
    xpt_hi[:] *= (1+eps[1])
    xeta_lo[:] -= eps[1]*mask
    xeta_hi[:] += eps[1]*mask
    xphi_lo[:] -= eps[2]*mask
    xphi_hi[:] += eps[2]*mask
    
    x_lo = x_lo.expand_as(deltas)
    x_hi = x_hi.expand_as(deltas)
    
    #x_lo = torch.cat([xpt_lo, xeta_lo, xphi_lo], axis=-1).expand_as(deltas)
    #x_hi = torch.cat([xpt_hi, xeta_hi, xphi_hi], axis=-1).expand_as(deltas)
    #print('x_lo', x_lo.shape)
    
    # note: not in-place
    xpt = xpt * (1+eps[0]*dpt)
    xeta = xeta + eps[1]*deta*mask
    xphi = xphi + eps[2]*dphi*mask
    
    xadv = torch.cat([xpt, xeta, xphi], axis=-1)
    #print('xadv', xadv.shape)
        
    #print('xpt   ', xpt.shape)
    
    #print('xadv  ', xadv.shape)
    
    y_repeat = y.view(-1,1).repeat(1,n_trials)
    #print('y_rep ', y_repeat.shape)
    
    history = []
    
    for i in range(n_iter):
        xadv.requires_grad = True
        
        preds = model(xadv)
        loss = model.loss_fn(preds.view(-1,2), y_repeat.view(-1)).mean()
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            dx = torch.sign(xadv.grad.detach())

            xpt, xeta, xphi = torch.chunk(xadv, 3, axis=-1)
            dpt, deta, dphi = torch.chunk(dx, 3, axis=-1)

            #xpt = xpt*(1+alpha[0]*dpt)
            #xeta = xeta + alpha[1]*deta*(mask>0)
            #xphi = xphi + alpha[2]*dphi*(mask>0)
            xpt *= (1+alpha[0]*dpt)
            xeta += alpha[1]*deta*mask
            xphi += alpha[2]*dphi*mask

            #xadv = torch.cat([xpt, xeta, xphi], axis=-1).detach()
            xadv = xadv.detach()
            
            # ensure the values are within the specified epsilon range
            #xadv = torch.where()
            #over = xadv>x_hi
            #under = xadv<x_lo
            #xadv[over] = x_hi[over]
            #xadv[under] = x_lo[under]
            xadv = torch.where(xadv>x_hi, x_hi, xadv)
            xadv = torch.where(xadv<x_lo, x_lo, xadv)
            
            if return_history:
                history.append(xadv.clone())
    
    
    if return_history:
        return history
    
    if not return_max:
        return xadv
    
    with torch.no_grad():
        preds = model(xadv)
        #print(preds.shape)
        losses = model.loss_fn(preds.view(-1,2), y_repeat.view(-1)).view_as(y_repeat)
        imax = losses.argmax(axis=-1)
        #print(imax.shape)
        #print(xadv.shape)
        return xadv[torch.arange(xadv.shape[0]),imax]
    
    
def fgsm_attack(model, x, y, epsilon=1e-1):
    xadv = x.clone()
    
    # note: these tensors are views into the original x object.
    xpt, xeta, xphi = torch.chunk(xadv, 3, axis=-1)
    
    # mask to suppress "missing" particles.
    mask = xpt>0
    
    # calculate the gradient of the model w.r.t. the *input* tensor:
    
    # first we tell torch that x should be included in grad computations
    xadv.requires_grad = True
    
    # then we just do the forward and backwards pass as usual:
    preds = model(xadv)
    loss = model.loss_fn(preds, y).mean()
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        # now we obtain the gradient of the input.
        # it has the same dimensions as the tensor xadv, and it "points"
        # in the direction of increasing loss values.
        dx = torch.sign(xadv.grad.detach())
        
        # so, we take a step in that direction!
        # However, in our problem, we have to take special care.
        # it makes sense to perturb eta and phi additively, but that doesn't
        # really work so well for pT.
        # Since the dynamic range of the pT variable is so huge, there's
        # no sensible choice of "step size". Instead, we should take a
        # "geometric" (i.e. multiplicative) step, rather than an arithmetic one.
        
        # first, split up the momentum coordinates so we can handle pT as a special case:
        dpt, deta, dphi = torch.chunk(dx, 3, axis=-1)
        
        # similarly for the xadv variable:
        xpt, xeta, xphi = torch.chunk(xadv, 3, axis=-1)
        
        # now, we make the necessary modifications *in-place*, so that
        # we are actually updating the xadv tensor behind the scenes.
        
        # pT is scaled up or down a fraction corresponding to the epsilon size:
        xpt *= (1 - epsilon*torch.sign(dpt))
        
        # each eta and phi takes a fixed size step (epsilon) in the direction (sign) of their gradient
        # we also mask out any changes to "missing" particles:
        xeta += epsilon*torch.sign(deta)*mask
        xphi += epsilon*torch.sign(dphi)*mask
        
        # now xadv contains the perturbed values; we can return it!
        return xadv.detach()