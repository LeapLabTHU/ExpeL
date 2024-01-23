import requests

from .wikienv import WikiEnv
from .wrappers import FeverWrapper
from envs.base import BaseEnv
from utils import parse_action


class FeverEnv(BaseEnv):
    def __init__(self, idx, max_steps: int = 6) -> None:
        self.env = FeverWrapper(WikiEnv(), split="dev")
        self.task = """fact extract and verification. The agent was given access to a Docstore API environment and a fact to verify. The agent can search for pages related to the fact, lookup keywords in the pages, and finish with an answer."""
        self.max_steps = max_steps
        self.idx = idx
        self.env_name = 'fever'

        self.reset()
    
    def reset(self):
        self.curr_step = 1
        self.question = self.env.reset(idx=self.idx).replace('Claim: ', '')
        self.key = self.env.data[self.idx][1]
        self.terminated = False
        return self.question

    def step(self, action):
        action_type, argument = parse_action(action)
        if action_type == 'Finish':
            self.terminated = True

        attempts = 0
        while attempts < 10:
            try:
                obs, self.reward, done, info = self.env.step(action[0].lower() + action[1:])
                self.curr_step += 1
                terminated = self.is_terminated()
                truncated = self.is_truncated()
                return obs, self.reward, terminated, truncated, self.curr_step
            except requests.exceptions.Timeout:
                attempts += 1

    def success_fn(self) -> bool:
        return self.reward == 1
