#  Train.
#  Set multistep input to 3: t-12, t-6, t
#  Output time range is fixed to 6 hours.

import math
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list,
        start_idx,
        end_idx,
        variables,
        out_variables,
        shuffle: bool = False,
        multi_dataset_training=False,
    ) -> None:
        super().__init__()
        start_idx = int(start_idx * len(file_list))
        end_idx = int(end_idx * len(file_list))
        file_list = file_list[start_idx:end_idx]
        self.file_list = [f for f in file_list if "climatology" not in f]
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle
        self.multi_dataset_training = multi_dataset_training

        assert self.variables == self.out_variables

    def __iter__(self):

        if self.shuffle:
            random.shuffle(self.file_list)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            if self.multi_dataset_training:
                num_nodes = int(os.environ.get("NODES", None))
                num_gpus_per_node = int(world_size / num_nodes)
                num_shards = num_workers_per_ddp * num_gpus_per_node
                rank = rank % num_gpus_per_node
            else:
                num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            data = np.load(path)
            yield {k: data[k] for k in self.variables}, self.variables, self.out_variables, data['hours'][:,0,0,0]


class Forecast(IterableDataset):
    def __init__(
        self, dataset: NpyReader, max_predict_range: int = 6, history_range: int = 12, hrs_each_step: int = 6
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.max_predict_range = max_predict_range
        self.history_range = history_range 
        self.hrs_each_step = hrs_each_step
        self.start_hour_per_year = [184080, 192864, 201624, 210384, 
                                    219144, 227928, 236688, 245448, 
                                    254208, 262992, 271752, 280512, 
                                    289272, 298056, 306816, 315576, 
                                    324336, 333120, 341880, 350640, 
                                    359400, 368184, 376944, 385704, 
                                    394464, 403248, 412008, 420768, 
                                    429528, 438312, 447072, 455832, 
                                    464592, 473376, 482136, 490896, 
                                    499656, 508440, 517200, 525960, 534720]
    
    def cal_time_embedding(self, hours):
        idx = np.searchsorted(self.start_hour_per_year, hours, 'right')
        start_hours = [self.start_hour_per_year[i-1] for i in idx]
        hour_in_year = hours - start_hours
        hour_in_year = torch.from_numpy(hour_in_year).float()
        return hour_in_year
    
    def __iter__(self):
        for data, variables, out_variables, hours in self.dataset:
            x = np.concatenate([data[k].astype(np.float32) for k in data.keys()], axis=1)
            x = torch.from_numpy(x)
            y = np.concatenate([data[k].astype(np.float32) for k in out_variables], axis=1)
            y = torch.from_numpy(y)
            
            time_embedding = self.cal_time_embedding(hours)

            assert self.max_predict_range % self.hrs_each_step == 0
            history_step = self.history_range // self.hrs_each_step
            predict_step = self.max_predict_range // self.hrs_each_step

            valid_len = x.shape[0] - history_step - predict_step
            inputs = torch.cat([x[i:i+valid_len] for i in range(history_step+1)], dim=1)
            outputs = x[1+history_step:,]
            yield inputs, outputs, variables, out_variables, time_embedding[history_step:-predict_step]



class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset, transforms: torch.nn.Module, output_transforms: torch.nn.Module, region_info = None):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.region_info = region_info

    def __iter__(self):
        for (inp, out, variables, out_variables, time_embedding) in self.dataset:
            assert inp.shape[0] == out.shape[0] # N,C,H,W
            assert inp.shape[1]//3 == out.shape[1]

            for i in range(inp.shape[0]):
                if self.region_info is not None:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), variables, out_variables, time_embedding[i], self.region_info
                else:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), variables, out_variables, time_embedding[i]


class ShuffleIterableDataset(IterableDataset):
    # set buffer_size in yaml to 1 to avoid shuffling.
    # Eliminate buffer_size to make sure the dataloader and slurm work correctly.
    # If buffer_size is to large, the dataloader will not work correctly or the slurm will kill the job due to the memory problem.
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()

if __name__ == '__main__':
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader

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

    def collate_fn(batch):
        inp = torch.stack([batch[i][0] for i in range(len(batch))])
        out = torch.stack([batch[i][1] for i in range(len(batch))])
        variables = batch[0][2]
        out_variables = batch[0][3]
        time_embedding = torch.stack([batch[i][4] for i in range(len(batch))])
        return (
            inp,
            out,
            [v for v in variables],
            [v for v in out_variables],
            time_embedding
        )

    normalize_mean = dict(np.load(os.path.join("/weatherbench2/WB2_240x121_npz", "normalize_mean.npz")))
    normalize_mean = np.concatenate([normalize_mean[var] for var in variables])
    normalize_std = dict(np.load(os.path.join("/weatherbench2/WB2_240x121_npz", "normalize_std.npz")))
    normalize_std = np.concatenate([normalize_std[var] for var in variables])
    output_transforms = transforms.Normalize(normalize_mean, normalize_std)
    transforms = transforms.Normalize(np.tile(normalize_mean, 3), np.tile(normalize_std, 3))

    train_list = [f"/weatherbench2/WB2_240x121_npz/train/{file}" for file in os.listdir("/weatherbench2/WB2_240x121_npz/train")]

    data_train = ShuffleIterableDataset(
        IndividualForecastDataIter(
            Forecast(
                NpyReader(
                    file_list=train_list,
                    start_idx=0,
                    end_idx=1,
                    variables=variables,
                    out_variables=variables,
                    shuffle=True,
                    multi_dataset_training=False,
                ),
                max_predict_range=6,
                hrs_each_step=6,
            ),
            transforms=transforms,
            output_transforms=output_transforms,
        ),
        buffer_size=1,
    )

    loader = DataLoader(
        data_train, 
        batch_size=4,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    iteration = iter(data_train)
    print(iteration.__next__())
    for i in range(10):
        _ = next(iteration)
        print(i)