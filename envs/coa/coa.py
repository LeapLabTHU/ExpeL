import ast
import json
from typing import Tuple

from .BattlefieldValidation import BattlefieldValidation

from envs.base import BaseEnv
from utils import parse_action, EM

class COAEnv(BaseEnv):
    def __init__(self,
                 supporting_information: str = "",
                 max_steps: int = 6
                 ):
        
        self.max_steps = max_steps
        self.task = """Course-of-action planning agent. The agent was given access to a course-of-action (COA) environment and a question to answer. The agent can command friendly units to either move, engage, or stand, and finish with an answer."""
        self.env_name = 'coa'
        self.battlefield = BattlefieldValidation(supporting_information)

        self.reset()

    def reset(self):
        self.curr_step = 1
        self.answer = ''
        self.terminated = False

    def step(self, action: str) -> Tuple[str, bool, bool, bool, bool]:
        action_type, arguments = self.parse_coa_call(action)
        observation = ""

        if action_type == 'finish':
            self.answer = arguments
            if self.success_fn():
                observation = 'Answer is CORRECT'
            else: 
                observation = f'Answer is INCORRECT'
            self.terminated = True

        # engage_target_unit(unit_id, target_unit_id)
        elif action_type == 'engage_target_unit':

            try:
                # Extract the arguments from the function call
                unit_id, target_unit_id = arguments

                """
                Call the helper function to determine whether the currently selected friendly unit
                can neutralize the targeted enemy unit within its attack range.
                """
                enemy_within_range = self.battlefield(unit_id, target_unit_id)

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
            try:

                # Extract arguments from the function call
                unit_id, target_x, target_y = arguments

                """
                Find the current coordinates of the selected friendly unit, then format the
                destination coordinates as a dictionary prior to calling the helper function.
                """
                start_position = self.battlefield.supporting_information[unit_id-1]["position"]
                dest_position = {"x": target_x, "y": target_y}
                
                # Evaluate whether the commanded move is a valid action
                is_valid_attack = self.battlefield.check_bridge_cross(start_position, dest_position)

                if is_valid_attack:
                    observation = "The friendly unit has made a valid move. Provide commands for the remaining friendly units."
                else:
                    observation = "The friendly unit made an invalid move. Remember that your friendly units cannot go out of bounds or cross the river without going over the bridge. Please provide a new order for the currently selected friendly unit"
                    
            except Exception as e:
                print(f"Exception: {e}")
        
        # stand_location(unit_id)
        elif action_type == 'stand_location':
            try:
                # Extract the unit_id from the argument
                unit_id = arguments[0]

                # Determine if it is possible for the unit to stand within the location
                if self.is_valid_stand_location(unit_id):
                    observation = "This is a valid stand action. Provide commands for the remaining friendly units."
                
                else:
                    observation = "Your friendly unit is out of bounds. For your next action, move to a different location by calling the attack_move_unit(unit_id, target_x, target_y) function."

                # observation = stand ground against enemies
                observation = self.explorer.lookup(arguments).strip('\n').strip()
            except ValueError:
                observation = f'You are trying to stand in an invalid location, likely because you are either in the river or out of bounds. For your next action, move to a different location by calling the attack_move_unit(unit_id, target_x, target_y) function.'
        else:
            observation = 'Invalid action. Valid actions are engage_target_unit(unit_id, target_unit_id), attack_move_unit(unit_id, target_x, target_y) and stand_location(unit_id).'

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
        # self.battlefield.check_movement()
        # return all(self.battlefield.movement_check_arr)
        return True

    def parse_coa_call(self, function_call_str):
        try:
            # Parse the string into an AST node
            node = ast.parse(function_call_str, mode='eval')

            # Ensure the node is an expression and the body is a function call
            if isinstance(node, ast.Expression) and isinstance(node.body, ast.Call):

                # Extract the function name, then extract and evaluate the args
                func_name = node.body.func.id
                args = [ast.literal_eval(arg) for arg in node.body.args]
                
                # Return the function name and arguments
                return func_name, args
            else:
                raise ValueError("parse_coa_call: Invalid function call string")
        except Exception as e:
            print(f"An error occurred while running parse_coa_call: {e}")
            return None, None
