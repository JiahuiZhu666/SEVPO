import os
import sys
sys.path.append('.')
import json
import re
from absl import app, flags
import wandb
from tqdm.auto import trange
import gymnasium as gym
from env.task_list import task_list
from env.toy import Toy
from model.wrappers import wrap_gym
from model.agent import SEVPO
from model.data.dsrl_datasets import DSRLDataset
from model.evaluation import evaluate, evaluate_toy
from ml_collections import config_flags, ConfigDict



FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 9, 'Choose env')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_string('project', '', 'project name for wandb')
flags.DEFINE_string('experiment_name', '', 'experiment name for wandb')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d

def load_diffusion_model(model_location, env):

    with open(os.path.join(model_location, 'config.json'), 'r') as file:
        cfg = to_config_dict(json.load(file))

    config_dict = dict(cfg['agent_kwargs'])
    model_cls = config_dict.pop("model_cls") 
    agent = globals()[model_cls].create(
        cfg['seed'], env.observation_space, env.action_space, **config_dict
    )

    def get_model_file():
        files = os.listdir(f"{model_location}")
        pickle_files = []
        for file in files:
            if file.endswith('.pickle'):
                pickle_files.append(file)
        numbers = {}
        for file in pickle_files:
            match = re.search(r'\d+', file)
            number = int(match.group())
            path = os.path.join(f"{model_location}", file)
            numbers[number] = path

        max_number = max(numbers.keys())
        max_path = numbers[max_number]
        return max_path
    
    model_file = get_model_file()
    new_agent = agent.load(model_file)
    print("Load Pretrained Model!")

    return new_agent


def call_main(details):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=details['agent_kwargs'])

    if details['env_name'] == 'Toy':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Toy"
        env = Toy(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    config_dict = dict(details['agent_kwargs'])
    config_dict['env_max_steps'] = env_max_steps

    model_cls = config_dict.pop("model_cls") 
    config_dict.pop("cost_scale") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )

    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])

        agent, info = agent.update(sample)

        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)

        if i % details['eval_interval'] == 0:
            agent.save(f"./results/{details['group']}/{details['experiment_name']}", save_time)
            save_time += 1
            if details['env_name'] == 'Toy':
                eval_info = evaluate_toy(agent, env, details['eval_episodes'])
            else:
                eval_info = evaluate(agent, env, details['eval_episodes'])
            if details['env_name'] != 'Toy':
                eval_info["normalized_return"], eval_info["normalized_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
            wandb.log({f"eval/{k}": v for k, v in eval_info.items()}, step=i)


def main(_):
    parameters = FLAGS.config
    if FLAGS.project != '':
        parameters['project'] = FLAGS.project
    parameters['env_name'] = task_list[FLAGS.env_id]
    parameters['ratio'] = FLAGS.ratio
    parameters['group'] = parameters['env_name']


    parameters['experiment_name'] = parameters['env_name']+'_'+str(parameters['agent_kwargs']['thres'])+'_'+str(parameters['agent_kwargs']['qcpi'])+'_' \
                                    + str(parameters['agent_kwargs']['safeloss'])+'_'+str(parameters['agent_kwargs']['unsafeloss'])+str(parameters['seed'])


    if parameters['env_name'] == 'Toy':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8

    print(parameters)

    if not os.path.exists(f"./results/{parameters['group']}/{parameters['experiment_name']}"):
        os.makedirs(f"./results/{parameters['group']}/{parameters['experiment_name']}")
    with open(f"./results/{parameters['group']}/{parameters['experiment_name']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)
    
    call_main(parameters)


if __name__ == '__main__':
    app.run(main)
