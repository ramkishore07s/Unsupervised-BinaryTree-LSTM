
# coding: utf-8

# In[76]:

import math

import torch
from torch import nn
from torch.nn import init

import basic


# In[6]:

class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.comp_linear.weight.data)
        init.constant_(self.comp_linear.bias.data, val=0)

    def forward(self, l=None, r=None):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """

        hl, cl = l
        hr, cr = r
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = self.comp_linear(hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl*(fl + 1).sigmoid() + cr*(fr + 1).sigmoid()
             + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


# In[80]:

def cosine_distance(hl, hr, hp):
    # return cosine similarity
    # add gumbell noise
    return torch.div(torch.sum(hl * hr, 2).float(), 
                     (torch.sqrt(torch.sum(hl * hl, 2).float()) * torch.sqrt(torch.sum(hr * hr, 2).float())))


# In[82]:

class TreeLSTM(nn.Module):
    def __init__(self, dim, gumbell_temperature=1.0, classifier=cosine_distance, training=False):
        
        super(TreeLSTM, self).__init__()
        self.dim = dim
        self.gumbell_temperature = gumbell_temperature
        self.compose_parent = BinaryTreeLSTMLayer(dim)
        self.score = classifier
        self.training = training


# In[84]:

def forward(self, input):
    h, c = input
    max_depth = h.size(1)
    for i in range(max_depth - 1):
        hl, hr = h[:, :-1, :], h[:, 1:, :]
        cl, cr = c[:, :-1, :], c[:, 1:, :]
        ph, pc = self.compose_parent((hl, cl), (hr, cr))

        # Compute scores for all adjacent nodes
        comp = self.score(hl, hr, ph)

        # Get probabilities
        if self.training: select_mask = basic.st_gumbel_softmax(comp, temperature=self.gumbell_temperature, mask=None)
        else: select_mask = basic.greedy_select(logits=comp, mask=None).float()
        
        # Compose binary masks based on probabilities
        select_mask_expanded = select_mask.unsqueeze(2).expand_as(hl)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask_expanded = (1 - select_mask_cumsum).unsqueeze(2).expand_as(hl)
        right_mask_expanded = (select_mask_cumsum - select_mask).unsqueeze(2).expand_as(hl)
        
        # Combine most probable nodes
        h = (select_mask_expanded * ph
             + left_mask_expanded * hl
             + right_mask_expanded * hr)
        c = (select_mask_expanded * pc
             + left_mask_expanded * cl
             + right_mask_expanded * cr)
            
    return h.squeeze(1), c.squeeze(1)


# In[ ]:

TreeLSTM.forward = forward

