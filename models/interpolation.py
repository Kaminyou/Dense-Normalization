# Copyright (c) 2024 Ming-Yang Ho, Che-Ming Wu, and Min-Sheng Wu
# All rights reserved.
#
# This source code is licensed under the AGPL License found in the
# LICENSE file in the root directory of this source tree.

import torch


class Interpolation3D:
    # input [C, 3, 3]
    # output [C, 512, 512]
    def __init__(self, channel, device="cuda"):
        self.channel = channel
        self.is_init = False
        self.device = device

    def init(self, size):
        if self.is_init:
            return

        self.size = size
        self.half_size = size // 2
        self.eps = 1e-7
        self.init_matrix()
        self.is_init = True

    def init_matrix(self):
        self.small_to_large = torch.arange(0.5, self.size + 0.5, 1).to(self.device)
        self.large_to_small = torch.arange(self.size - 0.5, 0, -1).to(self.device)

        self.top_left = (self.large_to_small * self.large_to_small.unsqueeze(0).T) / self.size / self.size  # noqa
        self.down_left = (self.large_to_small * self.small_to_large.unsqueeze(0).T) / self.size / self.size  # noqa
        self.top_right = (self.small_to_large * self.large_to_small.unsqueeze(0).T) / self.size / self.size  # noqa
        self.down_right = (self.small_to_large * self.small_to_large.unsqueeze(0).T) / self.size / self.size  # noqa

        self.top_left = self.top_left.contiguous()
        self.down_left = self.down_left.contiguous()
        self.top_right = self.top_right.contiguous()
        self.down_right = self.down_right.contiguous()

    def top_left_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        # [C, 1, 1] * [512, 512] -> [C, 512, 512]
        return top_left_value * self.top_left[-self.half_size:, -self.half_size:] + \
                top_right_value * self.top_right[-self.half_size:, -self.half_size:] + \
                down_left_value * self.down_left[-self.half_size:, -self.half_size:] + \
                down_right_value * self.down_right[-self.half_size:, -self.half_size:]

    def top_right_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return top_left_value * self.top_left[-self.half_size:, :self.half_size] + \
                top_right_value * self.top_right[-self.half_size:, :self.half_size] + \
                down_left_value * self.down_left[-self.half_size:, :self.half_size] + \
                down_right_value * self.down_right[-self.half_size:, :self.half_size]

    def down_left_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return top_left_value * self.top_left[:self.half_size, -self.half_size:] + \
                top_right_value * self.top_right[:self.half_size, -self.half_size:] + \
                down_left_value * self.down_left[:self.half_size, -self.half_size:] + \
                down_right_value * self.down_right[:self.half_size, -self.half_size:]

    def down_right_corner(self, top_left_value, top_right_value, down_left_value, down_right_value):
        return top_left_value * self.top_left[:self.half_size, :self.half_size] + \
                top_right_value * self.top_right[:self.half_size, :self.half_size] + \
                down_left_value * self.down_left[:self.half_size, :self.half_size] + \
                down_right_value * self.down_right[:self.half_size, :self.half_size]

    def _interpolation_mean_table(self, y0x0, y0x1, y0x2, y1x0, y1x1, y1x2, y2x0, y2x1, y2x2):
        # each input yaxb [C, 1, 1]
        table = torch.zeros((self.channel, self.size, self.size), device=self.device)
        table[:, :self.half_size, :self.half_size] = self.top_left_corner(y0x0, y0x1, y1x0, y1x1)
        table[:, :self.half_size, self.half_size:] = self.top_right_corner(y0x1, y0x2, y1x1, y1x2)
        table[:, self.half_size:, :self.half_size] = self.down_left_corner(y1x0, y1x1, y2x0, y2x1)
        table[:, self.half_size:, self.half_size:] = self.down_right_corner(y1x1, y1x2, y2x1, y2x2)

        return table

    def deal_with_inf(self, matrix_3x3):
        # to fill all the inf and nan with the middle values channel-wisely
        return torch.where(
            torch.logical_or(
                torch.isinf(matrix_3x3),
                torch.isnan(matrix_3x3),
            ),
            matrix_3x3[:, 1:2, 1:2],  # [C, 1, 1]
            matrix_3x3,  # [C, 3, 3]
        )

    def interpolation_mean_table(self, matrix_3x3):  # [C, 3, 3] be on the same device
        matrix_3x3 = self.deal_with_inf(matrix_3x3)
        matrix_3x3 = matrix_3x3.unsqueeze(-1).unsqueeze(-1)  # [C, 3, 3] -> [C, 3, 3, 1, 1]
        # matrix_3x3[:, 0, 0, :, :] => will be [C, 1, 1]
        return self._interpolation_mean_table(
            matrix_3x3[:, 0, 0, :, :],
            matrix_3x3[:, 0, 1, :, :],
            matrix_3x3[:, 0, 2, :, :],
            matrix_3x3[:, 1, 0, :, :],
            matrix_3x3[:, 1, 1, :, :],
            matrix_3x3[:, 1, 2, :, :],
            matrix_3x3[:, 2, 0, :, :],
            matrix_3x3[:, 2, 1, :, :],
            matrix_3x3[:, 2, 2, :, :],
        )

    def interpolation_std_table_inverse(self, matrix_3x3):  # [C, 3, 3] be on the same device
        matrix_3x3 = self.deal_with_inf(matrix_3x3)
        matrix_3x3 = matrix_3x3.unsqueeze(-1).unsqueeze(-1)  # [C, 3, 3] -> [C, 3, 3, 1, 1]
        matrix_3x3 = 1 / (matrix_3x3 + self.eps)
        return self._interpolation_mean_table(
            matrix_3x3[:, 0, 0, :, :],
            matrix_3x3[:, 0, 1, :, :],
            matrix_3x3[:, 0, 2, :, :],
            matrix_3x3[:, 1, 0, :, :],
            matrix_3x3[:, 1, 1, :, :],
            matrix_3x3[:, 1, 2, :, :],
            matrix_3x3[:, 2, 0, :, :],
            matrix_3x3[:, 2, 1, :, :],
            matrix_3x3[:, 2, 2, :, :],
        )
