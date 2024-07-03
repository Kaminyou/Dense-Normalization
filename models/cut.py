import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from models.base import BaseModel
from models.discriminator import Discriminator
from models.dn import (init_dense_instance_norm,
                       init_prefetch_dense_instance_norm,
                       not_use_dense_instance_norm, use_dense_instance_norm)
from models.generator import Generator
from models.projector import Head


class ContrastiveModel(BaseModel):
    # instance normalization can be different from the
    # one specified during training
    def __init__(self, config, normalization="in", parallelism=False):
        BaseModel.__init__(self, config)
        self.model_names = ["D_Y", "G", "H"]
        self.loss_names = ["G_adv", "D_Y", "G", "NCE"]
        self.visual_names = ["X", "Y", "Y_fake"]
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.loss_names += ["NCE_Y"]
            self.visual_names += ["Y_idt"]

        self.normalization = normalization
        # Discrimnator would not be used during inference,
        # so specification of instane normalization is not required
        self.D_Y = Discriminator().to(self.device)

        self.G = Generator(
            normalization=normalization,
            parallelism=parallelism,
        ).to(self.device)
        self.H = Head().to(self.device)

        self.opt_D_Y = optim.Adam(
            self.D_Y.parameters(),
            lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"],
            betas=(0.5, 0.999),
        )
        self.opt_G = optim.Adam(
            self.G.parameters(),
            lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"],
            betas=(0.5, 0.999),
        )
        self.opt_H = optim.Adam(
            self.H.parameters(),
            lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"],
            betas=(0.5, 0.999),
        )

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

        if self.config["TRAINING_SETTING"]["LOAD_MODEL"]:
            self.load_networks(self.config["TRAINING_SETTING"]["EPOCH"])

        lambda_lr = lambda epoch: 1.0 - max(  # noqa
            0, epoch - self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2
        ) / (self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2)
        self.scheduler_disc = lr_scheduler.LambdaLR(
            self.opt_D_Y, lr_lambda=lambda_lr
        )
        self.scheduler_gen = lr_scheduler.LambdaLR(
            self.opt_G, lr_lambda=lambda_lr
        )
        self.scheduler_mlp = lr_scheduler.LambdaLR(
            self.opt_H, lr_lambda=lambda_lr
        )

    def set_input(self, data):
        self.X = data["X_img"]
        self.Y = data["Y_img"]

    def forward(self):
        self.Y = self.Y.to(self.device)
        self.X = self.X.to(self.device)
        self.Y_fake = self.G(self.X)
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.Y_idt = self.G(self.Y)

    def analyze_feature_map(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake, feature_map = self.G.analyze_feature_map(X)
        return Y_fake, feature_map

    def inference(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake = self.G(X)
        return Y_fake

    def inference_with_anchor(self, X, y_anchor, x_anchor, padding=1, **kwargs):
        assert self.normalization == "dn"
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake = self.G.forward_with_anchor(
                X, y_anchor=y_anchor, x_anchor=x_anchor, padding=padding, **kwargs,
            )
        return Y_fake

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.D_Y, True)
        self.opt_D_Y.zero_grad()
        self.loss_D_Y = self.compute_D_loss()
        self.loss_D_Y.backward()
        self.opt_D_Y.step()

        # update G and H
        self.set_requires_grad(self.D_Y, False)
        self.opt_G.zero_grad()
        self.opt_H.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.opt_G.step()
        self.opt_H.step()

    def scheduler_step(self):
        self.scheduler_disc.step()
        self.scheduler_gen.step()
        self.scheduler_mlp.step()

    def compute_D_loss(self):
        # Fake
        fake = self.Y_fake.detach()
        pred_fake = self.D_Y(fake)
        self.loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        # Real
        self.pred_real = self.D_Y(self.Y)
        self.loss_D_real = self.mse(
            self.pred_real, torch.ones_like(self.pred_real)
        )

        self.loss_D_Y = (self.loss_D_fake + self.loss_D_real) / 2
        return self.loss_D_Y

    def compute_G_loss(self):
        fake = self.Y_fake
        pred_fake = self.D_Y(fake)
        self.loss_G_adv = self.mse(pred_fake, torch.ones_like(pred_fake))

        self.loss_NCE = self.calculate_NCE_loss(self.X, self.Y_fake)
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.Y, self.Y_idt)
            self.loss_NCE = (self.loss_NCE + self.loss_NCE_Y) * 0.5

        self.loss_G = self.loss_G_adv + self.loss_NCE
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.G(tgt, encode_only=True)
        feat_k, _ = self.G(src, encode_only=True, patch_ids=patch_ids_q)
        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
        loss = self.cross_entropy_loss(
            out, torch.arange(
                0, out.size(0), dtype=torch.long, device=self.device
            )
        )
        return loss

    def init_dense_instance_norm_for_whole_model(
        self,
        y_anchor_num,
        x_anchor_num,
    ):
        init_dense_instance_norm(
            self.G,
            y_anchor_num=y_anchor_num,
            x_anchor_num=x_anchor_num,
        )

    def init_prefetch_dense_instance_norm_for_whole_model(
        self,
        y_anchor_num,
        x_anchor_num,
    ):
        init_prefetch_dense_instance_norm(
            self.G,
            y_anchor_num=y_anchor_num,
            x_anchor_num=x_anchor_num,
        )

    def use_dense_instance_norm_for_whole_model(self):
        use_dense_instance_norm(self.G, padding=1)

    def not_use_dense_instance_norm_for_whole_model(self):
        not_use_dense_instance_norm(self.G)
