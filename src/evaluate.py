# -*- coding: utf-8 -*-

from play import mutual_play


def estimate_win_rate(network_a, network_b, device, conf):
    score_a, score_b = 0, 0
    for game in range(conf.NUM_GAMES // 2):
        # let network_a play black
        if mutual_play(network_a, network_b, device, conf):
            score_a += 1
        else:
            score_b += 1

        # let network_b play black
        if mutual_play(network_b, network_a, device, conf):
            score_b += 1
        else:
            score_a += 1

    return score_a / (score_a + score_b)
