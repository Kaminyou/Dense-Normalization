import torch.nn as nn

from models.dn import DenseInstanceNorm, PrefetchDenseInstanceNorm


def get_normalization_layer(
    out_channels,
    normalization='kin',
    parallelism=False,
    interpolate_mode='bilinear',
):
    if normalization == 'dn':
        if parallelism:
            return PrefetchDenseInstanceNorm(
                out_channels=out_channels,
            )
        return DenseInstanceNorm(
            out_channels=out_channels,
        )
    elif normalization == 'in':
        return nn.InstanceNorm2d(out_channels)
    else:
        raise NotImplementedError
