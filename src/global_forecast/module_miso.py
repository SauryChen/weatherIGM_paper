# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
# multi-input single-output, used for train.
from typing import Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

import sys
sys.path.append("src")

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.metrics import (
    lat_weighted_mse,
    lat_weighted_rmse_train,
    lat_weighted_rmse,
    lat_weighted_acc,
)

# from models.graph_cast_net import GraphCastNet
from models.model import model

class GlobalForecastModule(LightningModule):
    """Lightning module for global forecasting.

    Args:
        net : model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        net: nn.Module,
        train_cfg: dict = {},
        pretrained_path: str = "",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        # load weights only for the GAT layers for GAT ablation.
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
        new_state_dict = {}

        for name, param in checkpoint["state_dict"].items():
            if 'gat_layers' in name:
                print(name)
                new_state_dict[name] = param.clone().detach()
        self.load_state_dict(new_state_dict, strict=False)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r
    
    def set_hrs_each_step(self, delta_t):
        self.hrs_each_step = delta_t

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def training_step(self, batch: Any, batch_idx: int):
        num_outvar = batch['out'].shape[1]
        assert num_outvar == len(batch['out_variables']), "num_outvar must be equal to len(out_variables)"
        loss_dict, _ = self.net(
            batch['inp'], 
            batch['cons_inp'], 
            batch['time_embedding'],
            batch['out'], 
            # [lat_weighted_rmse_train], # will encounter nan.
            [lat_weighted_mse],
            lat=self.lat
        )
        loss_dict = loss_dict[0]

        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict["loss"]
        # import pdb; pdb.set_trace()
        # print(f"loss: {loss}")
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        log_postfix = f"{self.pred_range}_hours"
        all_loss_dicts = self.net.evaluate(
            batch['inp'], 
            batch['cons_inp'], 
            batch['time_embedding'],
            batch['out'], 
            batch['out_variables'],
            transform=self.denormalization,
            metrics=[lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix='6_hours',
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        log_postfix = "6_hours"
        all_loss_dicts = self.net.evaluate(
            batch['inp'], 
            batch['cons_inp'], 
            batch['time_embedding'],
            batch['out'], 
            batch['out_variables'],
            transform=self.denormalization,
            metrics=[lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)
        # import pdb; pdb.set_trace()
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.train_cfg['lr'],
                    "betas": (self.hparams.train_cfg['beta_1'], self.hparams.train_cfg['beta_2']),
                    "weight_decay": self.hparams.train_cfg['weight_decay'],
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.train_cfg['lr'],
                    "betas": (self.hparams.train_cfg['beta_1'], self.hparams.train_cfg['beta_2']),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.train_cfg['warmup_epochs'],
            self.hparams.train_cfg['max_epochs'],
            self.hparams.train_cfg['warmup_start_lr'],
            self.hparams.train_cfg['eta_min'],
        )

        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
