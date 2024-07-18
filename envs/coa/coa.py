import re
import string
from typing import Tuple
import time

from BattlefieldValidation import BattlefieldValidation

from envs.base import BaseEnv
from utils import parse_action, EM

class COAEnv(BaseEnv):
    def __init__(self,
                 question: str,
                 key: str,
                 max_steps: int = 6,
                 explorer: DocstoreExplorer = DocstoreExplorer(Wikipedia())):

        self.question = question
        self.key = key
        self.max_steps = max_steps
        self.explorer = explorer
        self.task = """multi-hop QA. The agent was given access to a Docstore API environment and a question to answer. The agent can search for pages related to the question, lookup keywords in the pages, and finish with an answer."""
        self.env_name = 'coa'

        self.reset()

    def reset(self):
        self.curr_step = 1
        self.answer = ''
        self.terminated = False

    def step(self, action: str) -> Tuple[str, bool, bool, bool, bool]:
        action_type, argument = parse_action(action)

        if action_type == 'Finish':
            self.answer = argument
            if self.success_fn():
                observation = 'Answer is CORRECT'
            else: 
                observation = f'Answer is INCORRECT'
            self.terminated = True

        # attack_move_unit(unit_id, target_x, target_y)
        elif action_type == 'Attack':
            while True:
                try:
                    # observation = attack an enemy and determine how this affects the situation
                    observation = self.explorer.search(argument).strip('\n').strip()
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)

        # engage_target_unit(unit_id, target_unit_id)
        elif action_type == 'Engage':
            try:
                # observation = engage an enemy and evaluate how this affect the plan 
                observation = self.explorer.lookup(argument).strip('\n').strip()
            except ValueError:
                observation = f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        
        # stand_location(unit_id)
        elif action_type == 'Stand':
            try:
                # observation = stand ground against enemies and determine how enemy casualties are handled 
                observation = self.explorer.lookup(argument).strip('\n').strip()
            except ValueError:
                observation = f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        else:
            observation = 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'

        self.curr_step += 1
        self.reward = self.success_fn()
        self.terminated = self.is_terminated()
        self.truncated = self.is_truncated()

        return observation, self.reward, self.terminated, self.truncated, self.curr_step
    
    """
    Returns that a military course of action is considered valid if "self.answer"
    only contains valid movements.
    """
    def success_fn(self) -> bool:
        field = BattlefieldValidation.BattlefieldValidation(self.answer)
        field.check_movement()
        return all(field.movement_check_arr)
        
