# weatherIGM
Source code for the model proposed in the paper:
**An Interpretable Weather Forecasting Model with Separately-Learned Dynamics and Physics Neural Networks** submitted to *Geophysical Research Letters (GRL)*

The code is based on pytorch_lightning for ddp, and torch_geometric for graph construction. The project builds on the codebase structure of [ClimaX](https://github.com/microsoft/ClimaX). The repository only contains the code for modeling training. The finetuning part can be modified based on this version.
