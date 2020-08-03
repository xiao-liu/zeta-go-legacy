# -*- coding: utf-8 -*-

import torch


class DefaultEvaluator:

    def __init__(self, network, device):
        self.network = network
        self.device = device

    def evaluate(self, features):
        self.network.eval()
        with torch.no_grad():
            features = features.to(self.device)
            return self.network(features)
