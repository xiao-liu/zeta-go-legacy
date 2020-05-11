# -*- coding: utf-8 -*-

import numpy as np

from go import BLACK, WHITE


def extract_features(node, conf):
    features = np.ndarray(
        shape=(conf.INPUT_CHANNELS, conf.BOARD_SIZE, conf.BOARD_SIZE),
        dtype=np.float32
    )
    for x in range(conf.BOARD_SIZE):
        for y in range(conf.BOARD_SIZE):
            features[conf.INPUT_CHANNELS - 1][x][y] = \
                1.0 if node.go.turn == BLACK else 0.0
    for i in range(conf.HISTORY_LENGTH):
        for x in range(conf.BOARD_SIZE):
            for y in range(conf.BOARD_SIZE):
                if node is None:
                    features[2 * i][x][y] = 0.0
                    features[2 * i + 1][x][y] = 0.0
                else:
                    features[2 * i][x][y] = \
                        1.0 if node.go.board.color(x, y) == BLACK else 0.0
                    features[2 * i + 1][x][y] = \
                        1.0 if node.go.board.color(x, y) == WHITE else 0.0
        if node is not None:
            node = node.parent
    return features
