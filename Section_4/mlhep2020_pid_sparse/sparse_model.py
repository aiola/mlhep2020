import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
from sparse_vd import LinearSVDO

class SparseNet(nn.Module):
    def __init__(self, input_dim, device, threshold=1.):
        super(SparseNet, self).__init__()
        self.fc1 = LinearSVDO(input_dim, 100, threshold=threshold, device=device)
        self.fc2 = LinearSVDO(100, 100, threshold=threshold, device=device)
        self.fc3 = LinearSVDO(100,  6, threshold=threshold, device=device)
        # verify that your model have threshold _Parameter_!
        # and that requires_grad=False
        self.threshold = Parameter(torch.as_tensor(threshold), requires_grad=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x