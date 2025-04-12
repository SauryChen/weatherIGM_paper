# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from scipy import stats


def get_lat_weights(lat):
    """Adapted from WeatherBench2 
        https://github.com/google-research/weatherbench2/blob/main/weatherbench2/metrics.py
    Args:
        lat: H
    """
    lat = np.deg2rad(lat)
    pi_over_2 = np.array([np.pi / 2], dtype = lat.dtype)
    bounds = np.concatenate([-pi_over_2, (lat[:-1] + lat[1:]) / 2, pi_over_2])
    weights = np.sin(bounds[1:]) - np.sin(bounds[:-1])
    weights /= np.mean(weights)
    return weights

def lat_weighted_mse(pred, y, vars, lat, mask=None):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [N, C, H, W]
    w_lat = get_lat_weights(lat)
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()
    return loss_dict

def lat_weighted_rmse_train(pred, y, vars, lat):
    # will cause NaN in train, but don't know why.
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]
    w_lat = get_lat_weights(lat)
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device) # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[var] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )
    loss_dict["loss"] = torch.sqrt(error * w_lat.unsqueeze(1)).mean(dim=1).mean()
    return loss_dict


def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]
    w_lat = get_lat_weights(lat)
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()
    loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict


def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    pred = transform(pred)
    y = transform(y)

    error = (pred - y) ** 2  # [B, V, H, W]
    w_lat = get_lat_weights(lat)
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )
    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])
    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    w_lat = get_lat_weights(lat)
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    clim = clim.to(device=y.device).unsqueeze(0) #(1,V,32,64)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def l2loss_sphere(solver, pred, y, relative=False, squared=True):
    loss = solver.integrate_grid((pred-y)**2, dimensionless = True).sum(dim = -1)
    if relative:
        loss = loss / solver.integrate_grid(y**2, dimensionless = True).sum(dim = -1)
    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()
    return loss

def loss_sfno(pred, y, vars, lat, mask=None):
    from torch_harmonics.examples import PdeDataset
    H = pred.shape[2]
    W = pred.shape[3]

    dt = 6*3600
    dt_solver = 150
    nsteps = int(dt/dt_solver)
    dataset = PdeDataset(dt = dt, nsteps = nsteps, dims = (H, W), device = pred.device, normalize = True)
    solver = dataset.solver.to(pred.device)

    error = (pred - y) ** 2  # [N, C, H, W]
    w_lat = get_lat_weights(lat)
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()
    
    loss_dict["loss"] = l2loss_sphere(solver, pred, y, relative=False, squared=True)
    return loss_dict