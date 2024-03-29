#!/usr/bin/env python

from RaceTrackLoader import RaceTracksDataset

import torch


def vectorDistanceLossFunction(preds, labels):
    pdist = torch.nn.PairwiseDistance(p=2)
    return pdist(preds, labels)


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

device = 'cpu'

dataset_basepath = "/media/micha/eSSD/datasets"
# dataset_basepath = "/home/micha/dev/datasets/droneracing"
dataset_basepath = "/data/datasets/pid_testing"
dataset_basename = "pid_evaluation"
# dataset_basename = "X4Gates_Circle_1"
# dataset_basename = "X4Gates_Circle_2"

dataset = RaceTracksDataset(dataset_basepath, dataset_basename, device=device)

dataset.exportRaceTracksAsTum()