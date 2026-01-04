import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class LinearFeatureBaseline(nn.Module):

    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff 

        self.weight = nn.Parameter(torch.Tensor(self.feature_size,),
                                   requires_grad=False)
        self.weight.data.zero_()
        self._eye = torch.eye(self.feature_size,
                              dtype=torch.float32,
                              device=self.weight.device)     

    @property
    def feature_size(self):
        return 2 * self.input_size + 4    # features for 2 variables

    def _feature(self, episodes):
        # ones = episodes.mask.unsqueeze(2)
        observations = episodes.observations
        device = observations.device
        T,B,obs_dim = observations.shape
        ones = torch.ones((T,B,1), device=device)
        # ones = torch.ones((T, B, 1), device=device)
        # time_step = torch.arange(T, device = device).view(-1, 1, 1) * ones / 100.0
        time_step = torch.arange(T, device=device).view(T, 1, 1).expand(T, B, 1) / 100.0
        

        return torch.cat([
            observations,
            observations ** 2,
            time_step,
            time_step ** 2,
            time_step ** 3,
            ones
        ], dim=2)   
    

    def fit(self, episodes):
        # sequence_length * batch_size x feature_size (8)
        featmat = self._feature(episodes).view(-1, self.feature_size)
        # sequence_length * batch_size x 1
        returns = episodes.returns.view(-1, 1)

        # Remove blank (all-zero) episodes that only exist because episode lengths vary
        flat_mask = episodes.mask.flatten()
        flat_mask_nnz = torch.nonzero(flat_mask)
        featmat = featmat[flat_mask_nnz].view(-1, self.feature_size)
        returns = returns[flat_mask_nnz].view(-1, 1)

        reg_coeff = self._reg_coeff
        
        XT_y = torch.matmul(featmat.t(), returns)   
        XT_X = torch.matmul(featmat.t(), featmat)   

        A_base = XT_X
        b       = XT_y
        coeffs  = None

        for i in range(5):
            reg = reg_coeff * (10 ** i)
            self._eye = self._eye.to(A_base.device)
            A = A_base + reg * self._eye
            x = torch.pinverse(A) @ b
            if not (torch.isnan(x).any() or torch.isinf(x).any()):
                coeffs = x
                break
        else:
            raise RuntimeError("Baseline fit still unstable after increasing regularization")

        self.weight.copy_(coeffs.flatten())

    def forward(self, episodes):
        features = self._feature(episodes)
        values = torch.mv(features.view(-1, self.feature_size), self.weight)
        return values.view(features.shape[:2])
