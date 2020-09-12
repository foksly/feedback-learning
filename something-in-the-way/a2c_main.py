import argparse
import json
import os
import pickle
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env import Environment, EnvFactory
from a2c import train
from a2c_agents import A2CAgent, A2CAttnAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a2c experiment')
    parser.add_argument('--env-config', help='Path to env config file')
    parser.add_argument('--meta', action='store_true')
    parser.add_argument('--hint-type', help='Type of the hint')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs of the experiment')
    parser.add_argument('--experiment-prefix', default='', help='Prefix for the experiment name')
    parser.add_argument('--experiment-config', 
                        default='/home/foksly/Documents/road-to-nips/feedback-learning/something-in-the-way/a2c/configs/default_config.json',
                        help='Experiment config')
    parser.add_argument('--logdir', help='Where to save logs')
    return parser.parse_args()


def get_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return config


def load_field_config(path):
    with open(path, 'rb') as file:
        config = pickle.load(file)
    return config


def get_experiment_name(args, add_date=True):
    env_name = args.env_config.split('/')[-1].split('.')[0] if args.env_config else 'meta'
    hint_type = args.hint_type if args.hint_type is not None else 'no_hint'
    date = datetime.now()
    
    experiment_name = '_'.join([env_name, hint_type, str(date.day), str(date.month)])
    if args.experiment_prefix:
        experiment_name = '_'.join([args.experiment_prefix, experiment_name])
    if args.meta:
        experiment_name = '_'.join(['meta', experiment_name])
    return experiment_name


def average_logs(logs):
    max_len = max(len(log) for log in logs)
    for log in logs:
        if len(log) < max_len:
            log.extend([log[-1] for _ in range(max_len - len(log))])
    logs_avg = np.mean(logs, axis=0)
    return logs_avg


def prepare_meta(experiment_config):
    factories = {}
    for t in ('train', 'test'):
        path = experiment_config[f'{t}_configs']['path']
        configs = sorted(os.listdir(path))
        configs = ['/'.join([path, config]) for config in configs]
        if 'size' in experiment_config[f'{t}_configs']:
            configs = configs[:experiment_config[f'{t}_configs']['size']]
        
        configs = [load_field_config(config) for config in configs]
        factories[t] = EnvFactory(configs)
    return factories['train'], factories['test']


def main():
    args = parse_args()
    experiment_config = get_config(args.experiment_config)
    
    if args.meta:
        train_factory, test_factory = prepare_meta(experiment_config)
    else:
        field_config = load_field_config(args.env_config)
        train_factory = EnvFactory(field_config)
        test_factory = None
    
    env = train_factory()
    experiment_config['state']['coord_range'] = np.prod(env.field.shape)
    experiment_config['state']['n_cols'] = env.field.shape[1]

    device = torch.device(experiment_config['train']['device'])
    train_params = experiment_config['train']
    experiment_name = get_experiment_name(args)
    save_path = experiment_config['train']['checkpoints_dir'] + experiment_name + '.pt'
    logdir = f'a2c/logs/{experiment_name}'
    if args.logdir is not None:
        logdir = args.logdir.rstrip('/') + '/' + experiment_name

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    reward_logs = []
    test_reward_logs = []
    for _ in tqdm(range(args.runs)):
        if args.hint_type is None:
            agent = A2CAgent(experiment_config['state'], n_actions=experiment_config['n_actions'],
                             receptive_field=env.receptive_field_size).to(device)
        elif args.hint_type.startswith('next_direction'):
            agent = A2CAgent(experiment_config['state'], n_actions=experiment_config['n_actions'],
                             hint_type=args.hint_type, hint_config=experiment_config['hint'], 
                             receptive_field=env.receptive_field_size).to(device)
        elif args.hint_type.startswith('attn'):
            agent = A2CAttnAgent(experiment_config['state'], n_actions=experiment_config['n_actions'],
                                 hint_type=args.hint_type, hint_config=experiment_config['hint'],
                                 attn_dim=experiment_config['attn_dim'], attn_type=experiment_config['attn_type'],
                                 receptive_field=env.receptive_field_size).to(device)


        optimizer = torch.optim.Adam(agent.parameters(), lr=experiment_config['train']['lr'])
        log = train(train_params['epochs'], 
                    train_params['n_agents'], 
                    train_factory, agent, optimizer, 
                    max_steps=train_params['max_steps'], 
                    hint_type=args.hint_type,
                    test_factory=test_factory,
                    device=device,
                    experiment_name=experiment_name,
                    save_path=save_path,
                    log_dir=logdir,
                    max_reward_limit=train_params['max_reward_limit'],
                    reward_log_freq=train_params['reward_log_freq'], 
                    plot_every=1)
        if test_factory is not None:
            train_log, test_log = log
            reward_logs.append(train_log)
            test_reward_logs.append(test_log)
        else:
            reward_logs.append(log)

    logs_avg = average_logs(reward_logs) if args.runs > 1 else reward_logs[0]
    if test_factory is not None:
        test_logs_avg = average_logs(test_reward_logs) if args.runs > 1 else test_reward_logs[0]
    
    writer = SummaryWriter(log_dir=logdir, filename_suffix=experiment_name)
    log_name = 'Reward' if test_factory is None else 'META/Train reward'
    for i, reward in enumerate(logs_avg):
        writer.add_scalar(log_name, reward, i)
    if test_factory is not None:
        for i, reward in enumerate(test_logs_avg):
            writer.add_scalar('META/Test reward', reward, i)
        


if __name__ == '__main__':
    main()
