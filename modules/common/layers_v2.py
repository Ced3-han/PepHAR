import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class _LayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-10):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super().__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )


LayerNorm = _LayerNorm


def glorot_uniform_af(x, gain=1.0):
    """
    initialize tensors the same as xavier_initializer in PyTorch, but the dimensions are different:
    In PyTorch:
    [feature_out, feature_in, n_head ...]
    In Jax:
    [... n_head, feature_in, feature_out]
    However, there is a feature in original Alphafold2 code that they use the Jax version initializer to initialize
    tensors like:
    [feature_in, n_head, feature_out]
    In this function, we keep this feature to initialize [feature_in, n_head, ..., feature_out] tensors
    """
    fan_in, fan_out = x.shape[-2:]
    if len(x.shape) > 2:
        receptive_field_size = np.prod(x.shape[:-2])
        fan_in *= receptive_field_size
        fan_out *= receptive_field_size
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    dev = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    nn.init.uniform_(x, -dev, dev)

    return x


class Linear(nn.Linear):

    def __init__(self,
                 feature_in: int,
                 feature_out: int,
                 initializer: str = 'linear',
                 use_bias: bool = True,
                 bias_init: float = 0., ):

        self.initializer = initializer
        self.bias_init = bias_init

        super().__init__(
            in_features=feature_in,
            out_features=feature_out,
            bias=use_bias
        )  # `reset_parameters` is called inside super.__init__

    def reset_parameters(self):
        if self.initializer == 'linear':
            glorot_uniform_af(self.weight, gain=1.0)
        elif self.initializer == 'relu':
            glorot_uniform_af(self.weight, gain=2.0)
        elif self.initializer == 'zeros':
            nn.init.zeros_(self.weight)

        if self.bias is not None:
            nn.init.constant_(self.bias, val=self.bias_init)


class OutProductMean(nn.Module):
    def __init__(self, n_feat=64, n_feat_out=128, n_feat_proj=32):
        super().__init__()
        self.layernormM = LayerNorm(n_feat)
        self.linear_a = Linear(n_feat, n_feat_proj)
        self.linear_b = Linear(n_feat, n_feat_proj)
        self.linear_z = Linear(n_feat_proj * n_feat_proj, n_feat_out, use_bias=True, initializer='zeros')

    def forward(self, M, M_mask=None):
        """
        Args:
            M:  (B, L, n_feat)
            M_mask: (B, L)
        """
        M = M.unsqueeze(1)  # (B, 1, L, n_feat)
        M = self.layernormM(M)
        a = self.linear_a(M) # (B,1,L,n_feat_proj)
        b = self.linear_b(M) # (B,1,L,n_feat_proj)
        if M_mask is not None:
            M_mask = M_mask.unsqueeze(1)  # (B, 1, L)
            M_mask = M_mask.unsqueeze(-1)  # (B, 1, L, *)
            a = M_mask * a # (B,1,L,n_feat_proj)
            b = M_mask * b # (B,1,L,n_faet_proj)
            # einsum: bsid,bsjd->bijd
            M_left = M_mask.permute(0, 3, 2, 1).contiguous().float()  # bsid -> bdis
            M_right = M_mask.permute(0, 3, 1, 2).contiguous().float()  # bsjd -> bdsj
            norm = torch.matmul(M_left, M_right).permute(0, 2, 3, 1).contiguous()  # bdij -> bijd
            # end einsum
        else:
            norm = float(M.shape[1])

        ## get outer product
        # einsum: bsid,bsje->bijde->bij(de) [a,b->O]
        # O = a[:, :, :, None, :, None] * b[:, :, None, :, None, :]   # bsi_d_, bs_j_e -> bsijde
        # O = O.sum(dim=1)    # bsijde -> bijde
        # O = O.reshape(*O.shape[:-2], -1)   # bijde -> bij(de)
        # end einsum

        ## get outer product
        O = torch.einsum('bsid,bsje->bijde', a, b).contiguous()
        O = O.reshape(*O.shape[:-2], -1)  # bijde -> bij(de)

        Z = self.linear_z(O) / (1e-3 + norm)  # bijf

        return Z


class NodeToPair(nn.Module):
    def __init__(self, n_feat=64, n_feat_out=128, n=4):
        super(NodeToPair, self).__init__()
        self.layernormM = LayerNorm(n_feat)
        self.linear_a = Linear(n_feat, n_feat * n, initializer='relu')
        self.linear_b = Linear(n_feat, n_feat * n, initializer='relu')
        self.linear_out = Linear(n_feat * n, n_feat_out, initializer='zeros')

    def forward(self, M, M_mask=None):
        """
        Args:
            M:  (B, L, n_feat)
            M_mask: (B, L)
        """
        M = self.layernormM(M)
        a = self.linear_a(M)
        b = self.linear_b(M)
        if M_mask is not None:
            M_mask = M_mask.unsqueeze(-1)
            a = M_mask * a
            b = M_mask * b
        o = a.unsqueeze(1) + b.unsqueeze(2)
        o = self.linear_out(F.relu(o))

        return o


class Attention(nn.Module):
    """
    Multi-Head Attention dealing with [batch_size1, batch_size2, len, dim] tensors
    """

    def __init__(self, q_dim, kv_dim, c, n_head, out_dim, gating=True):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.c = c
        self.n_head = n_head
        self.out_dim = out_dim
        self.gating = gating

        self.q_linear = Linear(q_dim, n_head * c, use_bias=False, initializer='zeros')
        self.k_linear = Linear(kv_dim, n_head * c, use_bias=False, initializer='zeros')
        self.v_linear = Linear(kv_dim, n_head * c, use_bias=False, initializer='zeros')

        if gating:
            self.gating_linear = Linear(q_dim, n_head * c, use_bias=True, initializer='zeros', bias_init=1.0)

        self.o_linear = Linear(n_head * c, out_dim, use_bias=True, initializer='zeros')

    def forward(self, q_data, kv_data, bias=None, nonbatched_bias=None):
        """
        :param q_data: [batch_size1, batch_size2, len_q, q_dim]
        :param kv_data: [batch_size1, batch_size2, len_kv, kv_dim]
        :param bias: None or [batch_size1, batch_size2, n_head, len_q, len_kv]
        :param nonbatched_bias: None or [batch_size1, n_head, len_q, len_kv]
        """
        n_head, c = self.n_head, self.c

        q = self.q_linear(q_data).reshape(*q_data.shape[:-1], n_head, c)  # bsqa -> bsq(hc) -> bsqhc
        q = q * (self.c ** (-0.5))
        k = self.k_linear(kv_data).reshape(*kv_data.shape[:-1], n_head, c)  # bska -> bsk(hc) -> bskhc
        v = self.v_linear(kv_data).reshape(*kv_data.shape[:-1], n_head, c)

        # einsum: bsqhc,bskhc->bshqk [q,k->logits]
        q_ = q.permute(0, 1, 3, 2, 4).contiguous()  # bsqhc -> bshqc
        k_ = k.permute(0, 1, 3, 4, 2).contiguous()  # bskhc -> bshck
        logits = torch.matmul(q_, k_)
        # logits = torch.einsum('bsqhc,bskhc->bshqk', q, k)

        if bias is not None:
            logits += bias
        if nonbatched_bias is not None:
            logits += nonbatched_bias.unsqueeze(1)

        weights = F.softmax(logits, dim=-1)
        # einsum: bshqk,bskhc->bsqhc
        # weights_ = weights.permute(0, 1, 3, 4, 2).unsqueeze(-1)  # bshqk -> bsqkh -> bsqkh_
        # v_ = v[:, :, None, :, :, :]   # bskhc -> bs_khc
        # weighted_avg = (weights_ * v_).sum(dim=3)   # bsqkh_, bs_khc -> bsqkhc -> bsqhc
        # end einsum
        weighted_avg = torch.einsum('bshqk,bskhc->bsqhc', weights, v)

        if self.gating:
            # gate_values = torch.einsum('bsqa,ahc->bsqhc', q_data, self.gating_weights) + self.gating_bias
            gate_values = self.gating_linear(q_data).unfold(-1, c, c)  # bsqa -> bsq(hc) -> bsqhc
            gate_values = torch.sigmoid(gate_values)
            weighted_avg *= gate_values

        output = self.o_linear(weighted_avg.reshape(*weighted_avg.shape[:-2], -1))  # bsqhc -> bsq(hc) -> bsqo
        return output


class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, d_pair, c=128):
        super().__init__()
        self.d_pair = d_pair
        self.c = c

        self.layernorm1 = LayerNorm(d_pair)
        self.left_projection = Linear(d_pair, c)
        self.left_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)
        self.right_projection = Linear(d_pair, c)
        self.right_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)

        self.output_gate = Linear(d_pair, d_pair, initializer='zeros', bias_init=1.)
        self.layernorm2 = LayerNorm(c)
        self.output_projection = Linear(c, d_pair, initializer='zeros', bias_init=0.)

    def forward(self, Z, Z_mask=None):
        """
        Args:
            Z:  (B, L, L, d_pair)
            Z_mask: (B, L, L)
        """
        Z = self.layernorm1(Z)
        left_proj_act = self.left_projection(Z)
        right_proj_act = self.right_projection(Z)
        if Z_mask is not None:
            Z_mask = Z_mask.unsqueeze(-1)
            left_proj_act = Z_mask * left_proj_act
            right_proj_act = Z_mask * right_proj_act
        left_proj_act *= torch.sigmoid(self.left_gate(Z))
        right_proj_act *= torch.sigmoid(self.right_gate(Z))

        g = torch.sigmoid(self.output_gate(Z))
        # einsum: bikd,bjkd->bijd
        # ab = (left_proj_act[:, :, None, :, :] * right_proj_act[:, None, :, :, :]).sum(dim=3)
        # bi_kd,b_jkd -> bijkd -> bijd
        # end einsum
        ab = torch.einsum('bikd,bjkd->bijd', left_proj_act, right_proj_act)

        Z = g * self.output_projection(self.layernorm2(ab))
        return Z


class TriangleMultiplicationIncoming(nn.Module):
    def __init__(self, d_pair, c=128):
        super().__init__()
        self.d_pair = d_pair
        self.c = c

        self.layernorm1 = LayerNorm(d_pair)
        self.left_projection = Linear(d_pair, c)
        self.left_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)
        self.right_projection = Linear(d_pair, c)
        self.right_gate = Linear(d_pair, c, initializer='zeros', bias_init=1.)

        self.output_gate = Linear(d_pair, d_pair, initializer='zeros', bias_init=1.)
        self.layernorm2 = LayerNorm(c)
        self.output_projection = Linear(c, d_pair, initializer='zeros', bias_init=0.)

    def forward(self, Z, Z_mask=None):
        """
        Args:
            Z:  (B, L, L, d_pair)
            Z_mask: (B, L, L)
        """
        Z = self.layernorm1(Z)
        left_proj_act = self.left_projection(Z)
        right_proj_act = self.right_projection(Z)
        if Z_mask is not None:
            Z_mask = Z_mask.unsqueeze(-1)
            left_proj_act = Z_mask * left_proj_act
            right_proj_act = Z_mask * right_proj_act
        left_proj_act *= torch.sigmoid(self.left_gate(Z))
        right_proj_act *= torch.sigmoid(self.right_gate(Z))

        g = torch.sigmoid(self.output_gate(Z))
        # einsum: bkjd,bkid->bijd
        # ab = (left_proj_act[:, :, None, :, :] * right_proj_act[:, :, :, None, :]).sum(dim=1)
        # bk_jd,bki_d -> bkijd -> bijd
        # end einsum
        ab = torch.einsum('bkjd,bkid->bijd', left_proj_act, right_proj_act)

        Z = g * self.output_projection(self.layernorm2(ab))
        return Z


class TriangleAttentionStartingNode(nn.Module):
    def __init__(self, d_pair, c=32, n_head=4):
        super().__init__()
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head

        self.layernorm1 = LayerNorm(d_pair)
        # _init_weights = torch.nn.init.normal_(torch.zeros([d_pair, n_head]), std=1.0/math.sqrt(d_pair))
        # self.linear_b_weights = nn.parameter.Parameter(data=_init_weights)
        self.linear_b = Linear(d_pair, n_head, use_bias=False)
        nn.init.normal_(self.linear_b.weight, std=1.0 / math.sqrt(d_pair))
        self.attention = Attention(q_dim=d_pair, kv_dim=d_pair, c=c, n_head=n_head, out_dim=d_pair, gating=True)

    def forward(self, Z, Z_mask=None):
        """
        Args:
            Z:  (B, L, L, d_pair)
            Z_mask: (B, L, L)
        """
        Z = self.layernorm1(Z)
        # b = torch.einsum('bqkc,ch->bhqk', Z, self.linear_b_weights)
        # einsum: bqkc,ch->bhqk
        b = self.linear_b(Z).permute(0, 3, 1, 2).contiguous()  # bqkc -> bqkh -> bhqk
        # end einsum

        if Z_mask is not None:  # padding mode
            padding_bias = (-1e4 * torch.logical_not(Z_mask))[:, :, None, None, :]
        else:
            padding_bias = None

        Z = self.attention(Z, Z, bias=padding_bias, nonbatched_bias=b)
        return Z


class TriangleAttentionEndingNode(nn.Module):
    def __init__(self, d_pair, c=32, n_head=4):
        super().__init__()
        self.d_pair = d_pair
        self.c = c
        self.n_head = n_head

        self.layernorm1 = LayerNorm(d_pair)
        # _init_weights = torch.nn.init.normal_(torch.zeros([d_pair, n_head]), std=1.0/math.sqrt(d_pair))
        # self.linear_b_weights = nn.parameter.Parameter(data=_init_weights)
        self.linear_b = Linear(d_pair, n_head, use_bias=False)
        nn.init.normal_(self.linear_b.weight, std=1.0 / math.sqrt(d_pair))
        self.attention = Attention(q_dim=d_pair, kv_dim=d_pair, c=c, n_head=n_head, out_dim=d_pair, gating=True)

    def forward(self, Z, Z_mask=None):
        """
        Args:
            Z:  (B, L, L, d_pair)
            Z_mask: (B, L, L)
        """
        Z = Z.transpose(-2, -3)
        Z_mask = Z_mask.transpose(-1, -2) if Z_mask is not None else None

        Z = self.layernorm1(Z)
        # b = torch.einsum('bqkc,ch->bhqk', Z, self.linear_b_weights)
        # einsum: bqkc,ch->bhqk
        b = self.linear_b(Z).permute(0, 3, 1, 2).contiguous()  # bqkc -> bqkh -> bhqk
        # end einsum

        if Z_mask is not None:  # padding mode
            padding_bias = (-1e4 * torch.logical_not(Z_mask))[:, :, None, None, :]
        else:
            padding_bias = None

        Z = self.attention(Z, Z, bias=padding_bias, nonbatched_bias=b)

        Z = Z.transpose(-2, -3)
        return Z


class NodeAttentionWithPairBias(nn.Module):
    def __init__(self, d_node, d_pair, c=32, n_head=8):
        super().__init__()
        self.c = c
        self.n_head = n_head
        self.layernormM = LayerNorm(d_node)
        self.layernormZ = LayerNorm(d_pair)
        # _init_weights = torch.nn.init.normal_(torch.zeros([d_pair, n_head]), std=1.0/math.sqrt(d_pair))
        # self.linear_b_weights = nn.parameter.Parameter(data=_init_weights, requires_grad=True)
        self.linear_b = Linear(d_pair, n_head, use_bias=False)
        nn.init.normal_(self.linear_b.weight, std=1.0 / math.sqrt(d_pair))

        self.attention = Attention(q_dim=d_node, kv_dim=d_node, c=c, n_head=n_head, out_dim=d_node, gating=True)

    def forward(self, M, Z, M_mask=None):
        """
        Args:
            M:  Nodewise features, (B, L, d_node)
            Z:  Pairwise features, (B, L, L, d_pair)
            M_mask: (B, L)
        """
        M = M.unsqueeze(1)  # (B, 1, L, d_node)
        ## Input projections
        M = self.layernormM(M)
        Z = self.layernormZ(Z)
        # einsum: bqkc,ch->bhqk
        b = self.linear_b(Z).permute(0, 3, 1, 2).contiguous()  # bqkc -> bqkh -> bhqk
        # end einsum

        if M_mask is not None:  # padding mode
            M_mask = M_mask.unsqueeze(1).float()  # (B, 1, L)
            padding_bias = (1e4 * (M_mask - 1.))[:, :, None, None, :]
        else:
            padding_bias = None

        M = self.attention(M, M, bias=padding_bias, nonbatched_bias=b)
        return M.squeeze(1)


class PairTransition(nn.Module):
    def __init__(self, d_pair, n=4):
        super().__init__()
        self.norm = LayerNorm(d_pair)
        self.linear1 = Linear(d_pair, n * d_pair, initializer='relu')
        self.linear2 = Linear(n * d_pair, d_pair, initializer='zeros')

    def forward(self, src):
        src = self.norm(src)
        src = self.linear2(F.relu(self.linear1(src)))
        return src


class NodeTransition(nn.Module):
    def __init__(self, d_node, n=4):
        super().__init__()
        self.norm = LayerNorm(d_node)
        self.linear1 = Linear(d_node, n * d_node, initializer='relu')
        self.linear2 = Linear(n * d_node, d_node, initializer='zeros')

    def forward(self, src):
        src = self.norm(src)
        src = self.linear2(F.relu(self.linear1(src)))
        return src


class DropoutRowwise(nn.Module):
    def __init__(self, p=0.25):
        super(DropoutRowwise, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, feat, dim=1):
        feat_size = list(feat.size())
        feat_size[dim] = 1
        x = torch.ones(feat_size).to(feat)
        return feat * self.dropout(x)