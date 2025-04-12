# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pytorch_lightning.cli import LightningCLI

import sys
sys.path.append("/weatherIGM/src")

from global_forecast.datamodule_miso import GlobalForecastDataModule
from global_forecast.module_miso import GlobalForecastModule

def main():
    cli = LightningCLI(
        model_class=GlobalForecastModule,
        datamodule_class=GlobalForecastDataModule,
        seed_everything_default=77,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "exit_on_error": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm

    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(lat=cli.datamodule.lat, lon=cli.datamodule.lon)
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    cli.model.set_hrs_each_step(cli.datamodule.hparams.hrs_each_step)
    cli.model.set_val_clim(cli.datamodule.val_clim)
    cli.model.set_test_clim(cli.datamodule.test_clim)
    
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()