import os
import torch
import math
import numpy as np


def get_lat_embed(embed_dir):
    lat = np.load(os.path.join(embed_dir, "lat.npy"))
    lat_embed = torch.from_numpy(np.cos(2*math.pi*lat / 180)).float()
    lat_embed = lat_embed.unsqueeze(-1) # (H,1)
    return lat, lat_embed

def get_lon_embed(embed_dir):
    lon = np.load(os.path.join(embed_dir, "lon.npy"))
    lon_embed = torch.from_numpy(np.sin(math.pi*((lon-180)/2)/180)).float()
    lon_embed = lon_embed.unsqueeze(0) # (1,W)
    return lon, lon_embed

def get_geo_embed(embed_dir):
    data = np.load(os.path.join(embed_dir, "constants.npz"))
    oro = torch.from_numpy(data['geopotential_at_surface']).float()
    oro = (oro - oro.mean()) / oro.std()
    lsm = torch.from_numpy(data['land_sea_mask']).float()
    lsm = (lsm - lsm.mean()) / lsm.std()
    slt = torch.from_numpy(data['soil_type']).float()
    slt = (slt - slt.mean()) / slt.std()

    return oro, lsm, slt
    

if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()
    embed_dir = '/weatherbench2/WB2_240x121_npz'
    lat_embed = get_lat_embed(embed_dir)
    lon_embed = get_lon_embed(embed_dir)
    oro, lsm, slt = get_geo_embed(embed_dir)
    print(lat_embed.shape, lon_embed.shape, oro.shape, lsm.shape, slt.shape)