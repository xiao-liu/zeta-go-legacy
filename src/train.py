# -*- coding: utf-8 -*-

import glog as log
import torch
import torch.optim as optim

from config import get_conf
from evaluate import estimate_win_rate
from example import ExamplePool
from network import ZetaGoNetwork


def learning_rate(step, conf):
    for threshold, lr in conf.LR_SCHEDULE:
        if threshold < 0 or step < threshold:
            return lr


def train(model_dir, conf_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_file is None:
        log.info('start training a new model')
        log.info('model_dir={}'.format(model_dir))
        log.info('conf_name={}'.format(conf_name))
        log.info('device={}'.format(device))

        conf = get_conf(conf_name)

        iteration = 0
        step = 0

        # randomly initialize a network and let it to be the current
        # best network
        network = ZetaGoNetwork(conf)
        best_network = ZetaGoNetwork(conf)
        best_network.load_state_dict(network.state_dict())
        network.to(device)
        best_network.to(device)

        # setup the optimizer
        # notice that the L2 regularization is implemented by
        # introducing a weight decay
        optimizer = optim.SGD(
            network.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=2 * conf.L2_REG)

        # create an example pool and fill it with examples
        log.info('initializing the example pool...')
        example_pool = ExamplePool(conf)
        example_pool.generate_examples(best_network, device)
        example_pool.shuffle()
    else:
        log.info('resume training from checkpoint {}'.format(checkpoint_file))
        log.info('device={}'.format(device))

        # load checkpoint and restore all the necessary states
        checkpoint = torch.load(checkpoint_file)

        conf = checkpoint['conf']

        iteration = checkpoint['iteration']
        step = checkpoint['step']

        network = ZetaGoNetwork(conf)
        network.load_state_dict(checkpoint['network'])
        best_network = ZetaGoNetwork(conf)
        best_network.load_state_dict(checkpoint['best_network'])
        network.to(device)
        best_network.to(device)

        optimizer = optim.SGD(
            network.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=2 * conf.L2_REG)
        optimizer.load_state_dict(checkpoint['optimizer'])

        example_pool = checkpoint['example_pool']

    running_loss = 0.0
    while iteration < conf.NUM_ITERATIONS:
        # train the model
        log.info('start iteration {}'.format(iteration))
        while example_pool.has_batch():
            # load a batch of examples
            features, pi, z = example_pool.load_batch(device)

            # set network to train mode
            network.train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            logp, v = network(features)
            loss = torch.mean(
                (z - v) ** 2 - torch.sum(pi * logp, dim=1, keepdim=True))

            # backward
            loss.backward()

            # set learning rate and optimize
            lr = learning_rate(step, conf)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % conf.CHECKPOINT_FREQUENCY == 0:
                log.info('[iter={}] checkpoint reached, step={}'
                         .format(iteration, step))

                # update best_network if the new network is stronger
                # notice that it is necessary to make a copy
                log.info('[iter={}] comparing current network ' +
                         'with best network...'.format(iteration))
                win_rate = estimate_win_rate(
                    network, best_network, device, conf)
                if win_rate > conf.WIN_RATE_MARGIN:
                    log.info('[iter={}] best network updated, win_rate={}'
                             .format(iteration, win_rate))
                    best_network.load_state_dict(network.state_dict())
                else:
                    log.info('[iter={}] best network not updated, win_rate={}'
                             .format(iteration, win_rate))

                # save model and print statistics
                running_loss /= conf.CHECKPOINT_FREQUENCY
                torch.save({
                    'conf': conf,
                    'iteration': iteration,
                    'step': step,
                    'example_pool': example_pool,
                    'best_network': best_network.state_dict(),
                    'network': network.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, '{}/checkpoint_{}.pt'.format(model_dir, step))
                log.info('[iter={}] checkpoint saved, running_loss={}'
                         .format(iteration, running_loss))
                running_loss = 0.0

        log.info('[iter={}] generating new examples for the next iteration...'
                 .format(iteration))
        example_pool.generate_examples(best_network, device)
        example_pool.shuffle()

        iteration += 1

    # training finished, save the final best network
    model_path = '{}/model.pt'.format(model_dir)
    torch.save({
        'conf': conf,
        'network': best_network.state_dict(),
    }, model_path)
    log.info('finished training, model saved to {}'.format(model_path))
