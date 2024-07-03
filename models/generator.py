import torch
import torch.nn as nn

from models.downsample import Downsample
from models.normalization import get_normalization_layer
from models.upsample import Upsample


class ResnetBlock(nn.Module):
    def __init__(
        self,
        features,
        normalization="in",
        parallelism=False,
    ):
        super().__init__()
        self.normalization = normalization
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            get_normalization_layer(
                features,
                normalization=normalization,
                parallelism=parallelism,
            ),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            get_normalization_layer(
                features,
                normalization=normalization,
                parallelism=parallelism,
            ),
        )

    def forward(self, x):
        return x + self.model(x)

    def forward_with_anchor(self, x, y_anchor, x_anchor, padding=1, **kwargs):
        assert self.normalization == "kin" or self.normalization == "dn"
        x_residual = x
        x = self.model[:2](x)
        x = self.model[2](
            x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
        )
        x = self.model[3:6](x)
        x = self.model[6](
            x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
        )
        return x_residual + x

    def analyze_feature_map(self, x):
        x = self.model[:2](x)
        feature_map1 = x
        x = self.model[2:6](x)
        feature_map2 = x
        x = self.model[6](x)
        return x, feature_map1, feature_map2


class GeneratorBasicBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        do_upsample=False,
        do_downsample=False,
        normalization="in",
        parallelism=False,
    ):
        super().__init__()

        self.do_upsample = do_upsample
        self.do_downsample = do_downsample
        self.normalization = normalization

        if self.do_upsample:
            self.upsample = Upsample(in_features)
        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size=3, stride=1, padding=1
        )
        self.instancenorm = get_normalization_layer(
            out_features,
            normalization=normalization,
            parallelism=parallelism,
        )
        self.relu = nn.ReLU(True)
        if self.do_downsample:
            self.downsample = Downsample(out_features)

    def forward(self, x):
        if self.do_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x

    def fordward_hook(self, x):
        if self.do_upsample:
            x = self.upsample(x)
        x_hook = self.conv(x)
        x = self.instancenorm(x_hook)
        x = self.relu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x_hook, x

    def forward_with_anchor(self, x, y_anchor, x_anchor, padding=1, **kwargs):
        assert self.normalization == "kin" or self.normalization == "dn"
        if self.do_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.instancenorm(
            x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
        )
        x = self.relu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x

    def analyze_feature_map(self, x):
        if self.do_upsample:
            x = self.upsample(x)
        feature_map = self.conv(x)
        x = self.instancenorm(feature_map)
        x = self.relu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x, feature_map


class Generator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        features=64,
        residuals=9,
        normalization="in",
        parallelism=False,
    ):
        super().__init__()
        self.residuals = residuals
        self.normalization = normalization

        self.reflectionpad = nn.ReflectionPad2d(3)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7),
            get_normalization_layer(
                features,
                normalization=normalization,
                parallelism=parallelism,
            ),
            nn.ReLU(True),
        )

        self.downsampleblock2 = GeneratorBasicBlock(
            features,
            features * 2,
            do_upsample=False,
            do_downsample=True,
            normalization=normalization,
            parallelism=parallelism,
        )
        self.downsampleblock3 = GeneratorBasicBlock(
            features * 2,
            features * 4,
            do_upsample=False,
            do_downsample=True,
            normalization=normalization,
            parallelism=parallelism,
        )

        self.resnetblocks4 = nn.Sequential(
            *[
                ResnetBlock(
                    features * 4,
                    normalization=normalization,
                    parallelism=parallelism,
                )
                for _ in range(residuals)
            ]
        )

        self.upsampleblock5 = GeneratorBasicBlock(
            features * 4,
            features * 2,
            do_upsample=True,
            do_downsample=False,
            normalization=normalization,
            parallelism=parallelism,
        )
        self.upsampleblock6 = GeneratorBasicBlock(
            features * 2,
            features,
            do_upsample=True,
            do_downsample=False,
            normalization=normalization,
            parallelism=parallelism,
        )

        self.block7 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, in_channels, kernel_size=7),
            nn.Tanh(),
        )

    def append_sample_feature(
        self,
        feature,
        return_ids,
        return_feats,
        mlp_id=0,
        num_patches=256,
        patch_ids=None,
    ):
        feature_reshape = feature.permute(0, 2, 3, 1).flatten(1, 2)  # B, F, C
        if patch_ids is not None:
            patch_id = patch_ids[mlp_id]
        else:
            patch_id = torch.randperm(feature_reshape.shape[1])
            patch_id = patch_id[: int(min(num_patches, patch_id.shape[0]))]
        x_sample = feature_reshape[:, patch_id, :].flatten(0, 1)

        return_ids.append(patch_id)
        return_feats.append(x_sample)

    def analyze_feature_map(self, x):
        feature_maps = {}
        x = self.reflectionpad(x)
        feature_map = self.block1[0](x)
        feature_maps["block1"] = feature_map.cpu().numpy()
        x = self.block1[1:](feature_map)

        x, feature_map = self.downsampleblock2.analyze_feature_map(x)
        feature_maps["downsampleblock2"] = feature_map.cpu().numpy()

        x, feature_map = self.downsampleblock3.analyze_feature_map(x)
        feature_maps["downsampleblock3"] = feature_map.cpu().numpy()

        for resnet_idx, resnetblock in enumerate(self.resnetblocks4):
            x, feature_map1, feature_map2 = resnetblock.analyze_feature_map(x)
            feature_map1_np = feature_map1.cpu().numpy()
            feature_map2_np = feature_map2.cpu().numpy()
            feature_maps[f"resnetblock4_{resnet_idx}_1"] = feature_map1_np
            feature_maps[f"resnetblock4_{resnet_idx}_2"] = feature_map2_np

        x, feature_map = self.upsampleblock5.analyze_feature_map(x)
        feature_maps["upsampleblock5"] = feature_map.cpu().numpy()

        x, feature_map = self.upsampleblock6.analyze_feature_map(x)
        feature_maps["upsampleblock6"] = feature_map.cpu().numpy()

        x = self.block7(x)
        return x, feature_maps

    def forward_with_anchor(self, x, y_anchor, x_anchor, padding=1, **kwargs):
        assert self.normalization == "kin" or self.normalization == "dn"
        x = self.reflectionpad(x)
        x = self.block1[0](x)
        x = self.block1[1](
            x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
        )
        x = self.block1[2](x)
        x = self.downsampleblock2.forward_with_anchor(
            x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
        )
        x = self.downsampleblock3.forward_with_anchor(
            x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
        )
        for resnetblock in self.resnetblocks4:
            x = resnetblock.forward_with_anchor(
                x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
            )
        x = self.upsampleblock5.forward_with_anchor(
            x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
        )
        x = self.upsampleblock6.forward_with_anchor(
            x, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
        )
        x = self.block7(x)
        return x

    def forward(self, x, encode_only=False, num_patches=256, patch_ids=None):
        if not encode_only:
            x = self.reflectionpad(x)
            x = self.block1(x)
            x = self.downsampleblock2(x)
            x = self.downsampleblock3(x)
            x = self.resnetblocks4(x)
            x = self.upsampleblock5(x)
            x = self.upsampleblock6(x)
            x = self.block7(x)
            return x

        else:
            return_ids = []
            return_feats = []
            mlp_id = 0

            x = self.reflectionpad(x)
            self.append_sample_feature(
                x,
                return_ids,
                return_feats,
                mlp_id=mlp_id,
                num_patches=num_patches,
                patch_ids=patch_ids,
            )
            mlp_id += 1

            x = self.block1(x)

            x_hook, x = self.downsampleblock2.fordward_hook(x)
            self.append_sample_feature(
                x_hook,
                return_ids,
                return_feats,
                mlp_id=mlp_id,
                num_patches=num_patches,
                patch_ids=patch_ids,
            )
            mlp_id += 1

            x_hook, x = self.downsampleblock3.fordward_hook(x)
            self.append_sample_feature(
                x_hook,
                return_ids,
                return_feats,
                mlp_id=mlp_id,
                num_patches=num_patches,
                patch_ids=patch_ids,
            )
            mlp_id += 1

            for resnet_layer_id, resnet_layer in enumerate(self.resnetblocks4):
                x = resnet_layer(x)
                if resnet_layer_id in [0, 4]:
                    self.append_sample_feature(
                        x,
                        return_ids,
                        return_feats,
                        mlp_id=mlp_id,
                        num_patches=num_patches,
                        patch_ids=patch_ids,
                    )
                    mlp_id += 1

            return return_feats, return_ids


if __name__ == "__main__":

    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    G = Generator()
    x = G(x)
    print(x.shape)

    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    G = Generator()
    feat_k_pool, sample_ids = G(x, encode_only=True, patch_ids=None)
    feat_q_pool, _ = G(x, encode_only=True, patch_ids=sample_ids)
    print(len(feat_k_pool))
