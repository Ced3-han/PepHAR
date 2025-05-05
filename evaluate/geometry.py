import numpy as np
import torch


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)

def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e

def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center    # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center    # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)    # (N, L, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat

### rotations
def _assert_equal(x, y):
    EPS = 1e-6
    if isinstance(x, (torch.Tensor, np.ndarray)):
        assert (abs(y - x) < EPS).all(), f'{x} != {y}'
    elif isinstance(x, (float, int)):
        assert abs(x - y) < EPS, f'{x} != {y}'
    else:
        raise TypeError('x must be torch.Tensor or np.ndarray')

def _unzip(x):
    assert isinstance(x, torch.Tensor)
    return [v for v in torch.movedim(x, -1, 0)]

def _zip(*xs):
    return torch.stack(xs, dim=-1)

def _zip_matrix(m):
    return torch.stack([torch.stack(v, axis=-1) for v in m], axis=-2)
    # return np.stack([np.stack(vec, axis=-1) for vec in mat], axis=-2)


def quaternion_to_rotation_matrix(Q):
    x, y, z, w = _unzip(Q)
    _assert_equal(x ** 2 + y ** 2 + z ** 2 + w ** 2, 1.0)
    # print(x, y, z, w)
    mat = [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ]
    return _zip_matrix(mat)


def rotation_matrix_to_quaternion(M):
    m00, m11, m22 = M[..., 0, 0], M[..., 1, 1], M[..., 2, 2]
    zero = torch.tensor(0)
    w = torch.sqrt(torch.maximum(zero, 1 + m00 + m11 + m22)) / 2
    x = torch.sqrt(torch.maximum(zero, 1 + m00 - m11 - m22)) / 2 * torch.sign(M[..., 2, 1] - M[..., 1, 2])
    y = torch.sqrt(torch.maximum(zero, 1 - m00 + m11 - m22)) / 2 * torch.sign(M[..., 0, 2] - M[..., 2, 0])
    z = torch.sqrt(torch.maximum(zero, 1 - m00 - m11 + m22)) / 2 * torch.sign(M[..., 1, 0] - M[..., 0, 1])
    _assert_equal(x ** 2 + y ** 2 + z ** 2 + w ** 2, 1.0)
    return _zip(x, y, z, w)


def axis_angle_to_rotation_matrix(axis, angle, angle_type='tri'):
    x, y, z = _unzip(axis)
    _assert_equal(x ** 2 + y ** 2 + z ** 2, 1.0)
    if angle_type == 'tri':
        c, s = _unzip(angle)
    elif angle_type == 'rad':
        c, s = torch.cos(angle), torch.sin(angle)
    _assert_equal(s ** 2 + c ** 2, 1.0)
    mat = [
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ]
    return _zip_matrix(mat)


def eular_to_rotation_matrix(x, y, z):
    cx, sx= torch.cos(x), torch.sin(x)
    cy, sy= torch.cos(y), torch.sin(y)
    cz, sz= torch.cos(z), torch.sin(z)
    mat = [
        [cy * cz, -cy * sz, sy],
        [cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx],
        [sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy]
    ]
    return _zip_matrix(mat)

def get_peptide_position():
    # Constants from GLY
    # psi_1 = psi_2 = 0, C1-N2 is 0/180' peptide-bond
    point_dict = {
        'N1': torch.tensor((-0.572, 1.337, 0.000)),
        'CA1': torch.tensor((0.000, 0.000, 0.000)),
        'C1': torch.tensor((1.517, -0.000, -0.000)),

        'N2': torch.tensor((2.1114, 1.1887, 0.0000)),
        'CA2': torch.tensor((3.5606, 1.3099, 0.0000)),
        'C2': torch.tensor((4.0913, -0.1112,  0.0000)),
    }
    # Constants from GLY
    return point_dict

