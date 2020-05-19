# -*- coding: utf-8 -*-

import glog as log
import numpy as np
import torch

from play import self_play
from resign import ResignManager


class ExamplePool:

    def __init__(self, conf):
        self.conf = conf

        # the list that stores examples
        self.examples = []

        # lengths[i] is the length (i.e., number of examples) of the
        # ith game
        self.lengths = []

        # the random permutation used to shuffle the examples
        self.permutation = None

        # records the current position when traversing the examples
        self.position = 0

        # resignation manager
        self.resign_mgr = ResignManager(conf)

    def generate_examples(self, network, device):
        for i in range(self.conf.GAMES_PER_ITERATION):
            new_examples = self_play(network, device, self.conf,
                                     self.resign_mgr)
            self.examples += new_examples
            self.lengths.append(len(new_examples))
            log.info('{} new examples generated'.format(len(new_examples)))

        # discard old examples when pool is full
        if len(self.lengths) > self.conf.EXAMPLE_POOL_SIZE:
            m = len(self.lengths) - self.conf.EXAMPLE_POOL_SIZE
            n = sum(self.lengths[:m])
            self.examples = self.examples[n:]
            self.lengths = self.lengths[m:]

    def shuffle(self):
        self.permutation = np.random.permutation(len(self.examples))
        self.position = 0

    def has_batch(self):
        return self.position + self.conf.BATCH_SIZE < len(self.examples)

    def load_batch(self, device):
        indices = self.permutation[
            self.position, self.position + self.conf.BATCH_SIZE]
        features = torch.stack(
            [torch.from_numpy(self.examples[i][0]) for i in indices]).to(device)
        pi = torch.stack(
            [torch.from_numpy(self.examples[i][1]) for i in indices]).to(device)
        z = torch.stack(
            [torch.from_numpy(self.examples[i][2]) for i in indices]).to(device)
        self.position += self.conf.BATCH_SIZE
        return features, pi, z
