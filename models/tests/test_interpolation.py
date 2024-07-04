# Copyright (c) 2024 Ming-Yang Ho, Che-Ming Wu
# All rights reserved.
#
# This source code is licensed under the AGPL License found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from ..interpolation import Interpolation3D


class Interpolation2D:
    # input [1] * 9
    # output [512, 512]
    def __init__(self):
        self.small_to_large = torch.arange(0.5, 512.5, 1)
        self.large_to_small = torch.arange(511.5, 0, -1)
        self.init_matrix()

    def init_matrix(self):
        self.top_left = (self.large_to_small * self.large_to_small.unsqueeze(0).T) / 512 / 512
        self.down_left = (self.large_to_small * self.small_to_large.unsqueeze(0).T) / 512 / 512
        self.top_right = (self.small_to_large * self.large_to_small.unsqueeze(0).T) / 512 / 512
        self.down_right = (self.small_to_large * self.small_to_large.unsqueeze(0).T) / 512 / 512

    def top_left_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return top_left_value * self.top_left[256:, 256:] + \
                top_right_value * self.top_right[256:, 256:] + \
                down_left_value * self.down_left[256:, 256:] + \
                down_right_value * self.down_right[256:, 256:]

    def top_right_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return top_left_value * self.top_left[256:, :256] + \
                top_right_value * self.top_right[256:, :256] + \
                down_left_value * self.down_left[256:, :256] + \
                down_right_value * self.down_right[256:, :256]

    def down_left_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return top_left_value * self.top_left[:256, 256:] + \
                top_right_value * self.top_right[:256, 256:] + \
                down_left_value * self.down_left[:256, 256:] + \
                down_right_value * self.down_right[:256, 256:]

    def down_right_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return top_left_value * self.top_left[:256, :256] + \
                top_right_value * self.top_right[:256, :256] + \
                down_left_value * self.down_left[:256, :256] + \
                down_right_value * self.down_right[:256, :256]

    def _interpolation_mean_table(self, y0x0, y0x1, y0x2, y1x0, y1x1, y1x2, y2x0, y2x1, y2x2):
        table = torch.zeros((512, 512))
        table[:256, :256] = self.top_left_corner(y0x0, y0x1, y1x0, y1x1)
        table[:256, 256:] = self.top_right_corner(y0x1, y0x2, y1x1, y1x2)
        table[256:, :256] = self.down_left_corner(y1x0, y1x1, y2x0, y2x1)
        table[256:, 256:] = self.down_right_corner(y1x1, y1x2, y2x1, y2x2)

        return table

    def _interpolation_std_table_inverse(
        self, y0x0, y0x1, y0x2, y1x0, y1x1, y1x2, y2x0, y2x1, y2x2,
    ):
        table = torch.zeros((512, 512))
        table[:256, :256] = self.top_left_corner(1 / y0x0, 1 / y0x1, 1 / y1x0, 1 / y1x1)
        table[:256, 256:] = self.top_right_corner(1 / y0x1, 1 / y0x2, 1 / y1x1, 1 / y1x2)
        table[256:, :256] = self.down_left_corner(1 / y1x0, 1 / y1x1, 1 / y2x0, 1 / y2x1)
        table[256:, 256:] = self.down_right_corner(1 / y1x1, 1 / y1x2, 1 / y2x1, 1 / y2x2)

        return table

    def interpolation_mean_table(self, table):
        return self._interpolation_mean_table(
            table[0, 0],
            table[0, 1],
            table[0, 2],
            table[1, 0],
            table[1, 1],
            table[1, 2],
            table[2, 0],
            table[2, 1],
            table[2, 2],
        )

    def interpolation_std_table_inverse(self, table):
        return self._interpolation_std_table_inverse(
            table[0, 0],
            table[0, 1],
            table[0, 2],
            table[1, 0],
            table[1, 1],
            table[1, 2],
            table[2, 0],
            table[2, 1],
            table[2, 2],
        )


def test_deal_with_inf():
    inter3d = Interpolation3D(channel=2, device="cpu")
    matrix_3x3 = torch.Tensor([
        [[1, 2, torch.inf], [4, 5, 6], [torch.nan, 8, 9]],
        [[torch.nan, 4, torch.nan], [3, 1, 7], [10, torch.inf, 9]],
    ])

    expected = torch.Tensor([
        [[1, 2, 5], [4, 5, 6], [5, 8, 9]],
        [[1, 4, 1], [3, 1, 7], [10, 1, 9]],
    ])

    check = inter3d.deal_with_inf(matrix_3x3)

    assert torch.equal(check, expected)


def test_mean_table_cpu():
    table = torch.Tensor([
        [[2, 3, 2], [2, 1, 2], [2, 2, 4]],
        [[5, 4, 3], [2, 1, 3], [6, 4, 1]],
    ])

    inter3d = Interpolation3D(channel=2, device="cpu")
    inter3d.init(size=512)

    inter2d = Interpolation2D()

    result = inter3d.interpolation_mean_table(table)
    for i in range(2):
        assert torch.allclose(result[i], inter2d.interpolation_mean_table(table[i]))


def test_std_table_cpu():
    table = torch.Tensor([
        [[2, 3, 2], [2, 1, 2], [2, 2, 4]],
        [[5, 4, 3], [2, 1, 3], [6, 4, 1]],
    ])

    inter3d = Interpolation3D(channel=2, device="cpu")
    inter3d.init(size=512)

    inter2d = Interpolation2D()

    result = inter3d.interpolation_std_table_inverse(table)
    for i in range(2):
        assert torch.allclose(result[i], inter2d.interpolation_std_table_inverse(table[i]))


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason='cuda is not available')
def test_mean_table_cuda():
    table = torch.Tensor([
        [[2, 3, 2], [2, 1, 2], [2, 2, 4]],
        [[5, 4, 3], [2, 1, 3], [6, 4, 1]],
    ])

    inter3d = Interpolation3D(channel=2, device="cuda")
    inter3d.init(size=512)

    inter2d = Interpolation2D()

    result = inter3d.interpolation_mean_table(table.to("cuda"))
    for i in range(2):
        assert torch.allclose(result[i], inter2d.interpolation_mean_table(table[i]).to("cuda"))


@pytest.mark.skipif(condition=not torch.cuda.is_available(), reason='cuda is not available')
def test_std_table_cuda():
    table = torch.Tensor([
        [[2, 3, 2], [2, 1, 2], [2, 2, 4]],
        [[5, 4, 3], [2, 1, 3], [6, 4, 1]],
    ])

    inter3d = Interpolation3D(channel=2, device="cuda")
    inter3d.init(size=512)

    inter2d = Interpolation2D()

    result = inter3d.interpolation_std_table_inverse(table.to("cuda"))
    for i in range(2):
        assert torch.allclose(
            result[i],
            inter2d.interpolation_std_table_inverse(table[i]).to("cuda"),
        )
