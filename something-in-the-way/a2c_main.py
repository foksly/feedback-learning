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
from a2c import A2CAgent, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a2c experiment')
    parser.add_argument('--env-config', help='Path to env config file')
    parser.add_argument('--test-env-config', help='Path to test env config file. META SETUP')
    parser.add_argument('--hint-type', help='Type of the hint')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs of the experiment')
    parser.add_argument('--experiment-name-prefix', default='', help='Prefix for the experiment name')
    parser.add_argument('--experiment-config', 
                        default='/home/foksly/Documents/road-to-nips/something-in-the-way/a2c/configs/default_config.json',
                        help='Experiment config')
    parser.add_argument('--logdir', help='Where to save logs')
    return parser.parse_args()


def get_config(path, env):
    with open(path, 'r') as file:
        config = json.load(file)
    config['state']['coord_range'] = np.prod(env.field.shape)
    config['state']['n_cols'] = env.field.shape[1]
    return config


def load_field_config(path):
    with open(path, 'rb') as file:
        config = pickle.load(file)
    return config


def get_experiment_name(args, add_date=True):
    env_name = args.env_config.split('/')[-1].split('.')[0]
    hint_type = args.hint_type if args.hint_type is not None else 'no_hint'
    date = datetime.now()
    
    experiment_name = '_'.join([env_name, hint_type, str(date.day), str(date.month)])
    if args.experiment_name_prefix:
        experiment_name = '_'.join([args.experiment_name_prefix, experiment_name])
    if args.test_env_config is not None:
        experiment_name = '_'.join(['meta', experiment_name])
    return experiment_name


def average_logs(logs):
    max_len = max(len(log) for log in logs)
    for log in logs:
        if len(log) < max_len:
            log.extend([log[-1] for _ in range(max_len - len(log))])
    logs_avg = np.mean(logs, axis=0)
    return logs_avg

def main():
    args = parse_args()

    field_config = load_field_config(args.env_config)
    make_env = EnvFactory(field_config)
    env = make_env()

    # test env for meta setup
    make_test_env = None
    if args.test_env_config is not None:
        test_field_config = load_field_config(args.test_env_config)
        make_test_env = EnvFactory(test_field_config)

    experiment_config = get_config(args.experiment_config, env)

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
            agent = A2CAgent(experiment_config['state'], receptive_field=env.receptive_field_size).to(device)
        else:
            agent = A2CAgent(experiment_config['state'], hint_type=args.hint_type, 
                            hint_config=experiment_config['hint'], receptive_field=env.receptive_field_size).to(device)

        optimizer = torch.optim.Adam(agent.parameters(), lr=experiment_config['train']['lr'])
        log = train(train_params['epochs'], 
                    train_params['n_agents'], 
                    make_env, agent, optimizer, 
                    max_steps=train_params['max_steps'], 
                    hint_type=args.hint_type,
                    make_test_env=make_test_env,
                    device=device,
                    experiment_name=experiment_name,
                    save_path=save_path,
                    log_dir=logdir,
                    max_reward_limit=train_params['max_reward_limit'],
                    reward_log_freq=train_params['reward_log_freq'], 
                    plot_every=1)
        if make_test_env is not None:
            train_log, test_log = log
            reward_logs.append(train_log)
            test_reward_logs.append(test_log)
        else:
            reward_logs.append(log)
        
    
    logs_avg = average_logs(reward_logs) if args.runs > 1 else reward_logs
    if make_test_env is not None:
        test_logs_avg = average_logs(test_reward_logs) if args.runs > 1 else test_reward_logs
    
    writer = SummaryWriter(log_dir=logdir, filename_suffix=experiment_name)
    log_name = 'Reward' if make_test_env is None else 'META/Train reward'
    for i, reward in enumerate(logs_avg):
        writer.add_scalar(log_name, reward, i)
    if make_test_env is not None:
        for i, reward in enumerate(test_logs_avg):
            writer.add_scalar('META/Test reward', reward, i)
        


if __name__ == '__main__':
    main()
