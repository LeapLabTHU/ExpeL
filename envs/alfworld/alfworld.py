import re
from typing import List, Dict, Any, Tuple
from envs.base import BaseEnv

import alfworld.agents.environment
from utils import get_env_name_from_gamefile

class AlfworldEnv(BaseEnv):
    def __init__(self,
                gamefile: str,
                config: Dict[str, Any],
                max_steps: int = 50,
                ):
        self.max_steps = max_steps
        self.gamefile = gamefile
        self.config = config
        self.main_env = getattr(alfworld.agents.environment, self.config.env.type)(self.config, train_eval=self.config.split)
        self.main_env.game_files = [self.gamefile]
        self.task = "housekeeper robot. The agent was placed in a household environment and a task to complete."
        self.env_name = get_env_name_from_gamefile(gamefile)

        self.reset()

    def reset(self):
        self.curr_step = 1
        self.answer = ''
        self.terminated = False
        self.reward = False
        self.is_exhausted = False
        self.env = self.main_env.init_env(batch_size=1)
        self.env.reset()
        self.last_action = None

    def step(self, action: str) -> Tuple[str, bool, bool, bool, int]:
        if action.startswith('put'):
            pattern = r'put (\w+\s*\d+) (?:in|on) (\w+\s*\d+)'
            match = re.match(pattern, action)
            if match is not None:
                action = 'put ' + match.group(1) + ' in/on ' + match.group(2)
        
        observation, reward, _ = self.alfworld_run(action)
        observation = observation.replace(' In it, you see nothing.', '').replace(', you see nothing', '')
        if self.last_action == action:
            self.truncated = True
            self.terminated = True
        
        self.last_action = action

        if reward:
                observation = 'Task is SOLVED.'
                self.terminated = True
        else:
            if self.is_truncated():
                observation = 'Max steps reached.'
            pass

        self.curr_step += 1
        self.terminated = self.is_terminated()
        self.truncated = self.is_truncated()
        self.reward = reward

        return observation, self.reward, self.terminated, self.truncated, self.curr_step

    def success_fn(self) -> bool:
        return self.reward
    
    def alfworld_run(self, action):
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_observation(observation[0]), info['won'][0], done[0]

        return observation, reward, done

def process_observation(obs):
    if obs.startswith('You arrive at loc '):
        obs = obs[obs.find('. ')+2:]    
    return obs