import os
import torch
import numpy as np
from typing import Optional
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

import torch.nn.functional as F
from itertools import product

import sys
sys.path.append("/weatherIGM/src")

from global_forecast.dataset_miso import (
    NpyReader,
    Forecast,
    IndividualForecastDataIter,
    ShuffleIterableDataset,
)

from utils.embedding import (
    get_lat_embed,
    get_lon_embed,
    get_geo_embed,
)


class GlobalForecastDataModule(LightningDataModule):
    """DataModule for global forecast data.

    Args:
        root_dir (str): Root directory for sharded data.
        variables (list): List of input variables.
        buffer_size (int): Buffer size for shuffling.
        out_variables (list, optional): List of output variables.
        history_range (int, optional): History range.
        predict_range (int, optional): Predict range.
        hrs_each_step (int, optional): Hours each step.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
    """

    def __init__(
        self,
        root_dir,
        variables,
        buffer_size,
        out_variables=None,
        history_range: int = 12,
        predict_range: int = 6,
        hrs_each_step: int = 6,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        if isinstance(out_variables, str):
            out_variables = [out_variables]
            self.hparams.out_variables = out_variables

        get_file_list = lambda partition: [os.path.join(root_dir, partition, file) 
                                           for file in os.listdir(os.path.join(root_dir, partition))]

        self.lister_train = get_file_list("train")
        self.lister_val = get_file_list("val")
        self.lister_test = get_file_list("test")

        normalize_mean = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[var] for var in variables])
        normalize_std = dict(np.load(os.path.join(self.hparams.root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[var] for var in variables])

        self.output_transforms = transforms.Normalize(normalize_mean, normalize_std)
        self.transforms = transforms.Normalize(np.tile(normalize_mean, 3), np.tile(normalize_std, 3))

        self.lat, self.lat_embed = get_lat_embed(root_dir)
        self.lon, self.lon_embed = get_lon_embed(root_dir)
        self.oro_embed, self.lsm_embed, self.slt_embed = get_geo_embed(root_dir)

        self.val_clim = self.get_climatology("val", out_variables)
        self.test_clim = self.get_climatology("test", out_variables)

        self.data_train: Optional[IterableDataset] = None
        self.data_val: Optional[IterableDataset] = None
        self.data_test: Optional[IterableDataset] = None


    def get_climatology(self, partition="train", variables=None):
        # I think climatology at val and test should also use the climatology in train ...
        clim_dict = dict(np.load(os.path.join(self.hparams.root_dir, "train", "climatology.npz")))
        if variables is None:
            variables = self.hparams.variables
        clim = np.concatenate([clim_dict[var] for var in variables]) 
        return torch.from_numpy(clim)

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ShuffleIterableDataset(
                IndividualForecastDataIter(
                    Forecast(
                        NpyReader(
                            file_list=self.lister_train,
                            start_idx=0,
                            end_idx=1,
                            variables=self.hparams.variables,
                            out_variables=self.hparams.out_variables,
                            shuffle=True,
                            multi_dataset_training=False,
                        ),
                        history_range=self.hparams.history_range,
                        max_predict_range=self.hparams.predict_range,
                        hrs_each_step=self.hparams.hrs_each_step,
                    ),
                    transforms=self.transforms,
                    output_transforms=self.output_transforms,
                ),
                buffer_size=self.hparams.buffer_size,
            )

            self.data_val = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_val,
                        start_idx=0,
                        end_idx=1,
                        variables=self.hparams.variables,
                        out_variables=self.hparams.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                    ),
                    history_range=self.hparams.history_range,
                    max_predict_range=self.hparams.predict_range,
                    hrs_each_step=self.hparams.hrs_each_step,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
            )

            self.data_test = IndividualForecastDataIter(
                Forecast(
                    NpyReader(
                        file_list=self.lister_test,
                        start_idx=0,
                        end_idx=1,
                        variables=self.hparams.variables,
                        out_variables=self.hparams.out_variables,
                        shuffle=False,
                        multi_dataset_training=False,
                    ),
                    history_range=self.hparams.history_range,
                    max_predict_range=self.hparams.predict_range,
                    hrs_each_step=self.hparams.hrs_each_step,
                ),
                transforms=self.transforms,
                output_transforms=self.output_transforms,
            )

    def collate_fn(self, batch):
        inp = torch.stack([i[0] for i in batch])
        out = torch.stack([i[1] for i in batch])
        variables = batch[0][2]
        out_variables = batch[0][3]
        
        B, _, H, W = inp.shape
        lat_emb = self.lat_embed.reshape(1,1,H,1).repeat(B, 1, 1, W)
        lon_emb = self.lon_embed.reshape(1,1,1,W).repeat(B, 1, H, 1)
        oro_emb = self.oro_embed.reshape(1,1,H,W).repeat(B, 1, 1, 1)
        lsm_emb = self.lsm_embed.reshape(1,1,H,W).repeat(B, 1, 1, 1)
        slt_emb = self.slt_embed.reshape(1,1,H,W).repeat(B, 1, 1, 1)

        time_embedding = torch.stack([batch[i][4] for i in range(len(batch))])
        cons_inp = torch.cat([lat_emb, lon_emb, oro_emb, lsm_emb, slt_emb], dim=1)
        
        return {
            "inp": inp,
            "cons_inp": cons_inp,
            "out": out,
            "variables": variables,
            "out_variables": out_variables,
            "time_embedding": time_embedding,
        }

    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn, 
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
        )
    
if __name__ == '__main__':

    variables = [
        "2m_temperature",
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential_50",
        "geopotential_250",
        "geopotential_500",
        "geopotential_600",
        "geopotential_700",
        "geopotential_850",
        "geopotential_925",
        "u_component_of_wind_50",
        "u_component_of_wind_250",
        "u_component_of_wind_500",
        "u_component_of_wind_600",
        "u_component_of_wind_700",
        "u_component_of_wind_850",
        "u_component_of_wind_925",
        "v_component_of_wind_50",
        "v_component_of_wind_250",
        "v_component_of_wind_500",
        "v_component_of_wind_600",
        "v_component_of_wind_700",
        "v_component_of_wind_850",
        "v_component_of_wind_925",
        "temperature_50",
        "temperature_250",
        "temperature_500",
        "temperature_600",
        "temperature_700",
        "temperature_850",
        "temperature_925",
        "specific_humidity_50",
        "specific_humidity_250",
        "specific_humidity_500",
        "specific_humidity_600",
        "specific_humidity_700",
        "specific_humidity_850",
        "specific_humidity_925",
    ]

    dataset = GlobalForecastDataModule(
        root_dir="/weatherbench2/WB2_240x121_npz",
        variables=variables,
        out_variables=variables,
        history_range=12,
        predict_range=6,
        hrs_each_step=6,
        batch_size=64,
        buffer_size=100,
        num_workers=1,
        pin_memory=False,
    )

    dataset.setup()

    iteration = iter(dataset.train_dataloader())
    for i in range(5):
        _ = next(iteration)
        print(i)
    print(_)

