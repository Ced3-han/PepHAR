from ._base import register_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import *

@register_model("seq_model")
class SeqModel(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.hidden_size = cfg.hid_dim
        self.embedding = nn.Embedding(21, cfg.hid_dim)

        self.lstm = nn.GRU(
            cfg.hid_dim, cfg.hid_dim,
            batch_first=True, num_layers=2,
            dropout=0.2
        )
        self.W_out = nn.Linear(cfg.hid_dim, 21)
        self.angle_out = nn.Linear(cfg.hid_dim,4)


    def forward(self, batch, max_rec_length=224, max_pep_length=32):
        embed,_ = self.lstm(self.embedding(batch['aa'])) # (B,L1+L2)
        embed = embed[:,max_rec_length:] # (B,L2,d)
        out = self.W_out(embed) # (B,L2,21)
        angle_out = self.angle_out(embed) # (B,L2,4)
        return out, angle_out

    def get_loss(self, batch):
        pred_logits,pred_type_angles = self.forward(batch) # (B,L2,21)
        loss = cross_entropy_with_mask(pred_logits, batch['label_types'], batch['label_types_mask'])
        acc = accuracy_with_mask(pred_logits, batch['label_types'], batch['label_types_mask'])
        label_angles,label_angles_mask = batch['label_angles'],batch['label_angles_mask']
        loss_regress = ((label_angles - pred_type_angles) ** 2 * label_angles_mask).sum() / label_angles_mask.sum()
        pred_abs = (torch.abs(label_angles - pred_type_angles) *label_angles_mask).sum() / label_angles_mask.sum()
        return {'clf_loss': loss, 'angle_loss': loss_regress}, pred_abs, acc

    def generate(self, batch):
        """
        :param batch: 这里是要从receptor出发逐渐填充的
        :return:
        """
        B = batch['rec_aa'].shape[0]
        rec_length = batch['rec_aa'].shape[-1] # (B,L1) -> (L1)
        pep_length = batch['pep_aa'].shape[-1] # (B,L2) -> (L2)
        S = torch.zeros_like(batch['pep_aa']) # (B,L1)
        for i in range(pep_length):
            if i==0:
                h_s, h = self.lstm(self.embedding(batch['rec_aa']))  # (B,L2,h*2), (2,B,h)
                h_s = h_s[:,-1] # (B,h*2)
            else:
                h_s, h = self.lstm(self.embedding(S[:, i - 1:i]),h)
            logits = self.W_out(h_s) # (B,21)
            prob = F.softmax(logits,dim=-1) # (B,21)
            S[:,i] = torch.multinomial(prob, num_samples=1).squeeze(-1)
        return S


    def _batch_to_feature(self, batch):
        return batch['aa']

    def sample(self,aa):
        embed,_ = self.lstm(aa) # (B,L1+L2)
        embed = embed[:,-1] # (B,L2,d)
        out = self.W_out(embed) # (B,21)
        angle_out = self.angle_out(embed) # (B,4)
        return out, angle_out



