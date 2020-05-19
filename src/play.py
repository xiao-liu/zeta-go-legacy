# -*- coding: utf-8 -*-

import numpy as np
import torch

from feature import extract_features
from go import BLACK, WHITE
from gui import GUI
from mcts import TreeNode, tree_search
from network import ZetaGoNetwork


def self_play(network, device, conf, resign_mgr):
    examples = []

    resign_enabled = resign_mgr.enabled()
    if resign_enabled:
        resign_threshold = resign_mgr.threshold()
    else:
        history = []

    # result undecided
    result = 0.0

    # create a search tree
    root = TreeNode(None, None, network, device, conf)

    previous_action = None
    t = 0
    while t < conf.MAX_GAME_LENGTH:
        # perform MCTS
        for i in range(conf.NUM_SIMULATIONS):
            tree_search(root, network, device, conf)

        # we follow AlphaGo's method to calculate the resignation value
        # notice that children with n = 0 are skipped by setting their
        # value to be -1.0 (w / n > -1.0 for children with n > 0)
        resign_value = max(
            map(lambda w, n: -1.0 if n == 0 else w / n, root.w, root.n))
        if not resign_enabled:
            history.append([resign_value, root.go.turn])
        elif -1.0 < resign_value <= resign_threshold:
            result = 1.0 if root.go.turn == WHITE else -1.0
            break

        # calculate the distribution of action selection
        # notice that illegal actions always have zero probability as
        # long as NUM_SIMULATION > 0
        if t < conf.EXPLORATION_TIME:
            # temperature tau = 1
            s = sum(root.n)
            pi = [x / s for x in root.n]
        else:
            # temperature tau -> 0
            m = max(root.n)
            p = [0 if x < m else 1 for x in root.n]
            s = sum(p)
            pi = [x / s for x in p]

        # save position, distribution of action selection and turn
        examples.append([
            extract_features(root, conf),
            np.array(pi, dtype=np.float32),
            np.array([root.go.turn], dtype=np.float32)])

        # choose an action
        action = np.random.choice(conf.NUM_ACTIONS, p=pi)

        # take the action
        root = root.children[action]

        # release memory
        root.parent.children = None

        t += 1

        # game terminates when both players pass
        if previous_action is not None \
                and previous_action == conf.PASS \
                and action == conf.PASS:
            break
        previous_action = action

    # calculate the scores if the result is undecided
    if result == 0.0:
        score_black, score_white = root.go.score()
        result = 1.0 if score_black > score_white else -1.0

    # add the history into resignation manager to update the threshold
    if not resign_enabled:
        resign_mgr.add(history, result)

    # update the the game winner from the perspective of each player
    for i in range(len(examples)):
        examples[i][2] *= result

    return examples


def mutual_play(network_black, network_white, device, conf):
    # create search trees for both players
    root_black = TreeNode(None, None, network_black, device, conf)
    root_white = TreeNode(None, None, network_white, device, conf)

    # black player goes first
    root = root_black
    network = network_black

    previous_action = None
    t = 0
    while t < conf.MAX_GAME_LENGTH:
        # both players perform MCTS, each one uses its own network
        for i in range(conf.NUM_SIMULATIONS):
            tree_search(root, network, device, conf)

        # calculate the distribution of action selection
        # temperature tau -> 0
        m = max(root.n)
        p = [0 if x < m else 1 for x in root.n]
        s = sum(p)
        pi = np.array([x / s for x in p], dtype=np.float32)

        # choose an action
        action = np.random.choice(conf.NUM_ACTIONS, p=pi)

        # take the action
        if root_black.children[action] is None:
            root_black.children[action] = \
                TreeNode(root_black, action, network_black, device, conf)
        root_black = root_black.children[action]
        if root_white.children[action] is None:
            root_white.children[action] = \
                TreeNode(root_white, action, network_white, device, conf)
        root_white = root_white.children[action]

        # release memory
        root_black.parent.children = None
        root_white.parent.children = None

        # switch to the other search tree
        root = root_white if root.go.turn == BLACK else root_black
        network = network_white if root.go.turn == BLACK else network_black

        t += 1

        # game terminates when both players pass
        if previous_action is not None \
                and previous_action == conf.PASS \
                and action == conf.PASS:
            break
        previous_action = action

    score_black, score_white = root.go.score()

    return score_black > score_white


def play_against_human(model_file, black_player):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_file)
    conf = model['conf']

    # load the network
    network = ZetaGoNetwork(conf)
    network.load_state_dict(model['network'])
    network.to(device)

    # create a search tree
    root = TreeNode(None, None, network, device, conf)

    gui = GUI(conf)

    human_turn = black_player == 'human'
    previous_action = None
    while True:
        if human_turn:
            # wait for human player's action
            action = gui.wait_for_action(root.go)
        else:
            # calculate computer's action
            gui.update_text('Computer is thinking...')

            # perform MCTS
            for i in range(conf.NUM_SIMULATIONS):
                tree_search(root, network, device, conf)

            # calculate the distribution of action selection
            # temperature tau -> 0
            m = max(root.n)
            p = [0 if x < m else 1 for x in root.n]
            s = sum(p)
            pi = np.array([x / s for x in p], dtype=np.float32)

            # choose an action
            action = np.random.choice(conf.NUM_ACTIONS, p=pi)

        # take the action
        if root.children[action] is None:
            root.children[action] = \
                TreeNode(root, action, network, device, conf)
        root = root.children[action]

        # release memory
        root.parent.children = None

        # update GUI
        gui.update_go(root.go)
        gui.update_text('Computer passes' if action == conf.PASS else '')

        # game terminates when both players pass
        if previous_action is not None \
                and previous_action == conf.PASS \
                and action == conf.PASS:
            black_score, white_score = root.go.score()
            winner = 'BLACK' if black_score > white_score else 'WHITE'
            gui.update_text('{} wins, {} : {}'.format(
                winner, black_score, white_score))
            gui.freeze()

        previous_action = action
        human_turn = not human_turn
