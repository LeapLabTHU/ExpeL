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
                 explorer: str = "DUMMY_VALUE"
                #  explorer: DocstoreExplorer = DocstoreExplorer(Wikipedia())
                 ):

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
        observation = ""

        if action_type == 'Finish':
            self.answer = argument
            if self.success_fn():
                observation = 'Answer is CORRECT'
            else: 
                observation = f'Answer is INCORRECT'
            self.terminated = True

        # engage_target_unit(unit_id, target_unit_id)
        elif action_type == 'engage_target_unit':

            REPLACE_WITH_HELPER_FUNCTION = lambda x: x+2

            try:

                """
                Call the helper function to determine whether the currently selected friendly unit
                can neutralize the targeted enemy unit within its attack range.
                """
                enemy_within_range = eval(REPLACE_WITH_HELPER_FUNCTION, "DUMMY VALUE")

                """
                Case 1: The enemy is within range

                If the enemy is within attack range, assume that the friendly unit neutralized the
                enemy. Provide good feedback to the model, and encourage it to continue planning
                successful engagements with the remaining friendly units.
                """
                if enemy_within_range:
                    observation = "You commanded the friendly unit to a successful engagement that neutralized the enemy target. Plan movements for the remaining friendly units, if any exist."

                    """
                    Case 2: The enemy is out of range

                    If the enemy is out of range, the friendly unit must move closer to take down the
                    enemy. Encourage the model to move the friendly unit closer to the targeted enemy.
                    Try utilizing a neutral tone to not discourage the model's ability to reason.
                    """

                else:
                    observation = "The engagement failed, as the targeted enemy unit is out of range. Please call the attack_move_unit(unit_id, target_x, target_y) function to move the currently selected friendly unit closer to the targeted enemy."

            except ValueError:
                observation = f'The helper function was unable to parse your engagement function call. Please try reformatting your engagement.'
        
        # attack_move_unit(unit_id, target_x, target_y)
        elif action_type == 'attack_move_unit':
            while True:
                try:
                    # observation = move to a position and provide feedback on
                    # whether this is a valid move or not
                    observation = self.explorer.search(argument).strip('\n').strip()
                    break
                except Exception as e:
                    print(f"Exception: {e}")
                    time.sleep(5)
        
        # stand_location(unit_id)
        elif action_type == 'Stand':
            try:
                # observation = stand ground against enemies
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
        
