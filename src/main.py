# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
from glob import glob
import os
import sys

from config import CONFIGURATIONS
from play import play_against_human
from train import train


def process_train():
    # parse arguments
    sub_parser = argparse.ArgumentParser(
        usage=(
            'python {0} train [--model_name MODEL_NAME] [--config CONFIG]\n' +
            '       ' +
            'python {0} train [-h]\n'
        ).format(sys.argv[0])
    )
    sub_parser.add_argument(
        '--model_name',
        type=str,
        default='',
        help='the name of the model, ' +
             'will use timestamp as name if not specified.')
    sub_parser.add_argument(
        '--config',
        type=str,
        default='19x19',
        help='the configuration for the training, ' +
             'must be one of the configurations defined in config.py ' +
             '(default: "19x19")')
    sub_args = sub_parser.parse_args(sys.argv[2:])

    if sub_args.config not in CONFIGURATIONS:
        print('configuration {} not found'.format(sub_args.config))
        exit(-1)

    model_name = datetime.now().strftime('%Y-%m-%d_%H%M%S') \
        if sub_args.model_name == '' else sub_args.model_name
    model_dir = os.path.abspath(os.path.join(
        os.getcwd(), '../models/{}'.format(model_name)))
    if os.path.exists(model_dir):
        print('directory {} already exists'.format(model_dir))
        exit(-1)
    os.makedirs(model_dir, exist_ok=True)

    train(model_dir, sub_args.config)


def process_resume():
    # parse arguments
    sub_parser = argparse.ArgumentParser(
        usage=(
            'python {0} resume <model_name> [--checkpoint CHECKPOINT]\n' +
            '       ' +
            'python {0} resume [-h]\n'
        ).format(sys.argv[0])
    )
    sub_parser.add_argument(
        'model_name',
        type=str,
        help='the name of the model to resume training')
    sub_parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='the name of the checkpoint file, ' +
             'will load the latest checkpoint if not specified')
    sub_args = sub_parser.parse_args(sys.argv[2:])

    model_dir = os.path.abspath(os.path.join(
        os.getcwd(), '../models/{}'.format(sub_args.model_name)))
    if not os.path.isdir(model_dir):
        print('directory {} not found'.format(model_dir))
        exit(-1)

    if sub_args.checkpoint == '':
        # load the latest checkpoint
        checkpoint_files = list(filter(
            lambda x: os.path.isfile(x),
            glob('{}/checkpoint_*.pt'.format(model_dir))))
        if len(checkpoint_files) == 0:
            print('no checkpoint file found in {}'.format(model_dir))
            exit(-1)
        checkpoint_file = max(checkpoint_files, key=os.path.getmtime)
    else:
        checkpoint_file = '{}/{}'.format(model_dir, sub_args.checkpoint)
        if not os.path.isfile(checkpoint_file):
            print('checkpoint file {} not found'.format(checkpoint_file))
            exit(-1)

    # resume training
    train(model_dir, None, checkpoint_file=checkpoint_file)


def process_play():
    # parse arguments
    sub_parser = argparse.ArgumentParser(
        usage=(
            'python {0} play <model_name> [--black_player BLACK_PLAYER]\n' +
            '       ' +
            'python {0} play [-h]\n'
        ).format(sys.argv[0])
    )
    sub_parser.add_argument(
        'model_name',
        type=str,
        help='the name of the model to play against')
    sub_parser.add_argument(
        '--black_player',
        type=str,
        default='human',
        help='the player who plays black and moves first, ' +
             'should be one of human/computer (default: human)')
    sub_args = sub_parser.parse_args(sys.argv[2:])

    model_file = os.path.abspath(os.path.join(
        os.getcwd(), '../models/{}/model.pt'.format(sub_args.model_name)))
    if not os.path.isfile(model_file):
        print('model file {} not found'.format(model_file))
        exit(-1)

    sub_args.black_player = sub_args.black_player.lower()
    if sub_args.black_player not in ('human', 'computer'):
        print('illegal black_player, set it to human')
        sub_args.black_player = 'human'

    play_against_human(model_file, sub_args.black_player)


def main():
    parser = argparse.ArgumentParser(
        usage=(
            'python {0} <command> [<args>]...\n       ' +
            'python {0} [-h]\n\n' +
            'Currently supported commands:\n' +
            '    train    Train a model\n' +
            '    resume   Resume training from a checkpoint\n' +
            '    play     Play Go with computer\n\n' +
            'Type "python {0} <command> -h" to show help message ' +
            'for each command.\n'
        ).format(sys.argv[0])
    )
    parser.add_argument(
        'command',
        type=str,
        help='the command to run, must be one of train/resume/play')
    args = parser.parse_args(sys.argv[1:2])

    if args.command == 'train':
        process_train()
    elif args.command == 'resume':
        process_resume()
    elif args.command == 'play':
        process_play()
    else:
        print('unrecognized command: {}'.format(args.command))
        exit(-1)


if __name__ == '__main__':
    main()
