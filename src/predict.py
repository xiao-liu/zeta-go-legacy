# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F

from feature import extract_features


# an implementation of dihedral group of order 8 (D4)
class Dihedral:

    @staticmethod
    def transform(x, trans, axes):
        if trans == 1:
            # identity
            return np.ascontiguousarray(x)
        elif trans == 2:
            # rotate counterclockwise by 90 degrees
            return np.ascontiguousarray(np.rot90(x, k=1, axes=axes))
        elif trans == 3:
            # rotate counterclockwise by 180 degrees
            return np.ascontiguousarray(np.rot90(x, k=2, axes=axes))
        elif trans == 4:
            # rotate counterclockwise by 270 degrees
            return np.ascontiguousarray(np.rot90(x, k=3, axes=axes))
        elif trans == 5:
            # reflect over x-axis
            return np.ascontiguousarray(np.flip(x, axis=axes[0]))
        elif trans == 6:
            # reflect over y-axis
            return np.ascontiguousarray(np.flip(x, axis=axes[1]))
        elif trans == 7:
            # reflect over x-axis and then rotate counterclockwise by
            # 90 degrees
            return np.ascontiguousarray(
                np.rot90(np.flip(x, axis=axes[0]), k=1, axes=axes))
        elif trans == 8:
            # reflect over y-axis and then rotate counterclockwise by
            # 90 degrees
            return np.ascontiguousarray(
                np.rot90(np.flip(x, axis=axes[1]), k=1, axes=axes))

    @staticmethod
    def inverse_transform(x, trans, axes):
        if trans == 1:
            return Dihedral.transform(x, 1, axes=axes)
        elif trans == 2:
            return Dihedral.transform(x, 4, axes=axes)
        elif trans == 3:
            return Dihedral.transform(x, 3, axes=axes)
        elif trans == 4:
            return Dihedral.transform(x, 2, axes=axes)
        elif trans == 5:
            return Dihedral.transform(x, 5, axes=axes)
        elif trans == 6:
            return Dihedral.transform(x, 6, axes=axes)
        elif trans == 7:
            return Dihedral.transform(x, 7, axes=axes)
        elif trans == 8:
            return Dihedral.transform(x, 8, axes=axes)


def predict(network, device, node, conf, random_trans=False):
    network.eval()
    with torch.no_grad():
        if random_trans:
            # uniform at random choose a Dihedral transformation and
            # apply it to the features
            trans = np.random.randint(1, 9)
            features = Dihedral.transform(
                extract_features(node, conf), trans, axes=(1, 2))
            features = torch.stack([torch.from_numpy(features)]).to(device)

            logp, v = network(features)

            # transform the distribution back
            p = F.softmax(logp, dim=1)
            p = p.cpu().numpy()[0]
            p_move, p_pass = p[:conf.BOARD_SIZE**2], p[conf.PASS]
            p_move = Dihedral.inverse_transform(
                np.reshape(p_move, (conf.BOARD_SIZE, conf.BOARD_SIZE)),
                trans, axes=(0, 1))
            p_move = np.reshape(p_move, conf.BOARD_SIZE**2)
            p = np.append(p_move, p_pass)

            v = v[0]
        else:
            features = extract_features(node, conf)
            features = torch.stack([torch.from_numpy(features)]).to(device)
            logp, v = network(features)
            p = F.softmax(logp, dim=1)
            p = p.cpu().numpy()[0]
            v = v[0]
    return p, v
