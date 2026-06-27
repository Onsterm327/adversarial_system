import torch
import torch.nn.functional as F
import math
from torchvision import transforms, datasets
import argparse
n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x


def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    return x
    # x = x * 2 ** n_bits
    # return torch.clamp(x, 0, 255).byte()

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def uniform_binning_correction(x, n_bits=8):
    """Replaces x^i with q^i(x) = U(x, x + 1.0 / 256.0).

    Args:
        x: 4-D Tensor of shape (NCHW)
        n_bits: optional.
    Returns:
        x: x ~ U(x, x + 1.0 / 256)
        objective: Equivalent to -q(x)*log(q(x)).
    """
    b, c, h, w = x.size()
    n_bins = 2 ** n_bits
    chw = c * h * w
    x += torch.zeros_like(x).uniform_(0, 1.0 / n_bins)

    objective = -math.log(n_bins) * chw * torch.ones(b, device=x.device)
    return x, objective


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, : C // 2, ...], tensor[:, C // 2 :, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


class transform_raw_to_grid(object):
    def __call__(self,tensor):
        tensor *= 255./256.
        tensor += 1./512.
        return tensor

class transform_grid_to_raw(object):
    def __call__(self, tensor):
        tensor *= 256./255.
        tensor -= 1./510.
        return tensor

def ebm_to_raw(dset):
    transform = transform_grid_to_raw()
    return transform

def raw_to_ebm(dset):
    transform = transform_raw_to_grid()
    return transform

def parse_args_diffusion():
    parser = argparse.ArgumentParser(description='eval_generative_defense')
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--debug", type=bool, default=False, help="")
    parser.add_argument("--log_path", type=str, default='logs', help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--sanity_check", type=bool, default=False, help="")
    parser.add_argument("--show_plots", type=bool, default=False, help="")
    parser.add_argument("--dataset", type=str, default='cifar10', help="")
    parser.add_argument("--cls_model", type=str, default='resnet18_cifar10', help="")

    ## generative params ##
    parser.add_argument("--first_step", type=int, default=140, help="")
    parser.add_argument("--arch", type=str, default='ddim', help="")
    parser.add_argument("--timesteps", type=int, default=100, help="")

    ## adv params ##
    parser.add_argument("--adv_num_steps", type=int, default=20, help="")
    parser.add_argument("--adv_epsilon", type=float, default=0.03137254, help="")
    parser.add_argument("--adv_step_size", type=float, default=0.00392156, help="")
    parser.add_argument("--adv_attack_type", type=str, default='white', help="")
    parser.add_argument("--adv_threat_model", type=str, default='linf', help="")
    parser.add_argument("--adv_EOT", type=int, default=20, help="")


    args = parser.parse_args()
    args.adv_step_size = 2.5 * (args.adv_epsilon / args.adv_num_steps)
    return args