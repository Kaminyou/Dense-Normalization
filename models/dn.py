import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.interpolation import Interpolation3D


class DenseInstanceNorm(nn.Module):
    def __init__(
        self,
        out_channels: int,
        affine: bool = True,
        device: str = 'cuda',
        interpolate_mode: str = 'bilinear',
    ):
        super(DenseInstanceNorm, self).__init__()

        if interpolate_mode not in ['bilinear']:
            raise ValueError('interpolate_mode supports bilinear only')

        # if use normal instance normalization during evaluation mode
        self.normal_instance_normalization = False

        # if collecting instance normalization mean and std
        # during evaluation mode'
        self.collection_mode = False

        self.out_channels = out_channels
        self.device = device
        self.interpolate_mode = interpolate_mode

        self.interpolation3d = Interpolation3D(channel=out_channels, device=device)
        if affine:
            self.weight = nn.Parameter(
                torch.ones(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)
            self.bias = nn.Parameter(
                torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)

    def init_collection(self, y_anchor_num: int, x_anchor_num: int) -> None:
        # TODO: y_anchor_num => grid_height, x_anchor_num => grid_width
        self.y_anchor_num = y_anchor_num
        self.x_anchor_num = x_anchor_num
        self.mean_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )
        self.std_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )

    def pad_table(self, padding: int = 1) -> None:
        # modify
        # padded table shape inconsisency
        # TODO: Don't permute the dimensions

        pad_func = nn.ReplicationPad2d((padding, padding, padding, padding))
        self.padded_mean_table = pad_func(
            self.mean_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]
        self.padded_std_table = pad_func(
            self.std_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]

    def forward_normal(self, x: torch.Tensor) -> torch.Tensor:
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x = (x - x_mean) / x_std  # * self.weight + self.bias
        return x

    def forward(
        self,
        x: torch.Tensor,
        y_anchor: t.Optional[int] = None,
        x_anchor: t.Optional[int] = None,
        padding: int = 1,
    ) -> torch.Tensor:
        # TODO: Do not reply on self.training
        if self.training or self.normal_instance_normalization:
            _, _, h, w = x.shape
            self.interpolation3d.init(size=h)
            return self.forward_normal(x)

        else:
            if y_anchor is None:
                raise ValueError('y_anchor is None')

            if x_anchor is None:
                raise ValueError('x_anchor is None')

            if self.collection_mode:
                _, _, h, w = x.shape
                self.interpolation3d.init(size=h)
                x_std, x_mean = torch.std_mean(x, dim=(2, 3))  # [B, C]
                # x_anchor, y_anchor = [B], [B]
                # table = [H, W, C]
                # update std and mean to corresponing coordinates
                self.mean_table[y_anchor, x_anchor] = x_mean
                self.std_table[y_anchor, x_anchor] = x_std
                x_mean = x_mean.unsqueeze(-1).unsqueeze(-1)
                x_std = x_std.unsqueeze(-1).unsqueeze(-1)

                x = (x - x_mean) / x_std * self.weight + self.bias

            else:
                if x.shape[0] != 1:
                    raise ValueError('only support batch size = 1')

                top = y_anchor
                down = y_anchor + 2 * padding + 1
                left = x_anchor
                right = x_anchor + 2 * padding + 1
                x_mean = self.padded_mean_table[
                    :, :, top:down, left:right
                ]  # 1, C, H, W
                x_std = self.padded_std_table[
                    :, :, top:down, left:right
                ]  # 1, C, H, W

                x_mean = self.interpolation3d.interpolation_mean_table(
                    x_mean[0],
                ).unsqueeze(0)
                x_std = self.interpolation3d.interpolation_std_table_inverse(
                    x_std[0],
                ).unsqueeze(0)

                x = (x - x_mean) * x_std * self.weight + self.bias
            return x


def not_use_dense_instance_norm(model: nn.Module) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.collection_mode = False
            layer.normal_instance_normalization = True


def init_dense_instance_norm(
    model: nn.Module, y_anchor_num: int, x_anchor_num: int,
) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.collection_mode = True
            layer.normal_instance_normalization = False
            layer.init_collection(
                y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num
            )


def use_dense_instance_norm(model, padding: int = 1) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, DenseInstanceNorm):
            layer.pad_table(padding=padding)
            layer.collection_mode = False
            layer.normal_instance_normalization = False


class PrefetchDenseInstanceNorm(nn.Module):
    def __init__(
        self,
        out_channels: int,
        affine: bool = True,
        device: str = 'cuda',
        interpolate_mode: str = 'bilinear',
    ):
        super(PrefetchDenseInstanceNorm, self).__init__()

        # if use normal instance normalization during evaluation mode

        # if collecting instance normalization mean and std
        # during evaluation mode'

        if interpolate_mode not in ['bilinear']:
            raise ValueError('interpolate_mode supports bilinear and bicubic only')

        self.out_channels = out_channels
        self.device = device
        self.interpolate_mode = interpolate_mode

        self.interpolation3d = Interpolation3D(channel=out_channels, device=device)
        if affine:
            self.weight = nn.Parameter(
                torch.ones(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)
            self.bias = nn.Parameter(
                torch.zeros(size=(1, out_channels, 1, 1), requires_grad=True)
            ).to(device)
        self.pad_func = nn.ReplicationPad2d((1, 1, 1, 1))

    def init_collection(self, y_anchor_num: int, x_anchor_num: int) -> None:
        # TODO: y_anchor_num => grid_height, x_anchor_num => grid_width
        self.y_anchor_num = y_anchor_num
        self.x_anchor_num = x_anchor_num
        self.mean_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )
        self.std_table = torch.zeros(
            y_anchor_num, x_anchor_num, self.out_channels
        ).to(
            self.device
        )
        self.pad_table()

    def pad_table(self, padding: int = 1) -> None:
        # modify
        # padded table shape inconsisency
        # TODO: Don't permute the dimensions

        pad_func = nn.ReplicationPad2d((padding, padding, padding, padding))

        self.padded_mean_table = pad_func(
            self.mean_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]
        self.padded_std_table = pad_func(
            self.std_table.permute(2, 0, 1).unsqueeze(0)
        )  # [H, W, C] -> [C, H, W] -> [N, C, H, W]

    def forward_normal(self, x: torch.Tensor) -> torch.Tensor:
        x_std, x_mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        x = (x - x_mean) / x_std  # * self.weight + self.bias
        return x

    def forward(
        self,
        x: torch.Tensor,
        y_anchor: int = None,
        x_anchor: int = None,
        padding: int = 1,
        pre_y_anchor: t.Optional[t.List[int]] = None,
        pre_x_anchor: t.Optional[t.List[int]] = None,
    ) -> torch.Tensor:
        n, _, h, w = x.shape
        real_x, pre_x = torch.split(x, (1, n - 1), dim=0)

        self.interpolation3d.init(size=h)

        if pre_y_anchor is not None and pre_x_anchor is not None:
            pre_x_std, pre_x_mean = torch.std_mean(pre_x, dim=(2, 3))  # [B, C]

            for i, (sub_pre_y_anchor, sub_pre_x_anchor) in enumerate(zip(pre_y_anchor, pre_x_anchor)):  # noqa
                if sub_pre_y_anchor == -1:
                    continue
                self.mean_table[sub_pre_y_anchor, sub_pre_x_anchor] = pre_x_mean[i]
                self.std_table[sub_pre_y_anchor, sub_pre_x_anchor] = pre_x_std[i]

            pre_x_mean = pre_x_mean.unsqueeze(-1).unsqueeze(-1)
            pre_x_std = pre_x_std.unsqueeze(-1).unsqueeze(-1)

            pre_x = (pre_x - pre_x_mean) / pre_x_std * self.weight + self.bias

        if y_anchor != -1 and x_anchor != -1:
            top = y_anchor
            left = x_anchor

            down = y_anchor + 2 * padding + 1
            right = x_anchor + 2 * padding + 1

            self.pad_table()
            x_mean = self.padded_mean_table[
                :, :, top:down, left:right
            ]  # [1, C, H, W]
            x_std = self.padded_std_table[
                :, :, top:down, left:right
            ]  # [1, C, H, W]
            x_mean = x_mean.squeeze(0)  # [1, C, H, W] -> [C, H, W]
            x_std = x_std.squeeze(0)  # [1, C, H, W] -> [C, H, W]
            x_mean_expand = x_mean[:, 1, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
            x_std_expand = x_std[:, 1, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 3)
            x_mean = torch.where(x_mean == 0, x_mean_expand, x_mean)
            x_std = torch.where(x_std == 0, x_std_expand, x_std)
            x_mean = self.interpolation3d.interpolation_mean_table(x_mean).unsqueeze(0)
            x_std = self.interpolation3d.interpolation_std_table_inverse(x_std).unsqueeze(0)

            real_x = (real_x - x_mean) * x_std * self.weight + self.bias

        x = torch.cat((real_x, pre_x), dim=0)
        return x


def init_prefetch_dense_instance_norm(
    model: nn.Module, y_anchor_num: int, x_anchor_num: int,
) -> None:
    for _, layer in model.named_modules():
        if isinstance(layer, PrefetchDenseInstanceNorm):
            layer.init_collection(
                y_anchor_num=y_anchor_num, x_anchor_num=x_anchor_num
            )
