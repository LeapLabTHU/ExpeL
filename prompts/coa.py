from typing import Tuple, Dict, Any, List
import pickle
import re

from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    ChatMessage,
)

FEWSHOTS = [""""""]

REFLECTION_FEWSHOTS = [""""""]

SYSTEM_INSTRUCTION = """You are a military commander assistant. Your users are military commanders and your role is to help them develop a military courses of action (COA). The military commanders will inform you the mission objective, terrain information, and available friendly and hostile assets before you start developing the COA. Given this information, you will develop a number of courses of action (as specified by the commander) so they can iterate on them with you and pick their favorite one. For each COA to be complete, every friendly unit needs to be assigned one command from the list below. Remember, Hostile units cannot be assigned any command! 
1) attack_move_unit(unit_id, target_x, target_y): commands friendly unit to move to target (x, y) coordinate in the map engaging hostile units in its path.
2) engage_target_unit(unit_id, target_unit_id): commands friendly unit to engage with hostile target unit, which is located at the target (x, y) coordinate in the map. If out of range, friendly unit will move to the target unit location before engaging.
3) stand_location(unit_id): commands friendly unit to stand ground at current location and engage any hostile units that are in range.
Remember, it is of vital importance that all friendly units are given commands. All generated COAs should be aggregated in a single JSON object following the template below:

```
{{
"coa_id_0": {{
	"overview": <describes overall strategy for this COA, explain why it is feasible (the COA can accomplish the mission within the established
time, space, and resource limitations), acceptable (the COA must balance cost and risk with advantage gained), suitable (the COA can accomplish the mission objective), and distinguishable (each COA must differ significantly from the others)."",
	"name": "<name that summarizes this particular COA>",
	"task_allocation": [
		{{"unit_id": 4295229441, "unit_type": "Mechanized infantry", "alliance": "Friendly", "position": {{"x": 14.0, "y": 219.0}}, "command": "attack_move_unit(4295229441, 35.0, 41.0); engage_target_unit(4295229441, 3355229433)"}},
		{{"unit_id": 4299948033, "unit_type": "Aviation", "alliance": "Friendly", "position": {{"x": 10.0, "y": 114.0}}, "command": "
engage_target_unit(4295229441, 3355229433) "}},
		{{"unit_id": 4382918273, "unit_type": "Armor", "alliance": "Friendly", "position": {{"x": 10.0, "y": 114.0}}, "command": "
stand_location(4295229441) "}}
	<continues for all friendly units, every single one of them: all friendly units need commands which can be a chain of an unlimited amount of the three allowed commands (only chains of two are shown above)>
	]
}}
}}
```

Here's additional military information that might be useful when generating COAs:

- The forms of maneuver are envelopment, flank attack, frontal attack, infiltration, penetration, and turning movement. Commanders use these forms of maneuver to orient on the enemy, not terrain.
- The four primary offensive tasks are movement to contact, attack, exploitation, and pursuit. While it is convenient to talk of them as different tasks, in reality they flow readily from one to another.
- There are three basic defensive tasks - area defense, mobile defense, and retrograde.


The mission is taking place in the following map/terrain:

The map is split in two major portions (west and east sides) by a river that runs from north to south right in the middle of a 200 x 200 map. There are two bridges in order to cross the river. Bridges names and their coordinates are as follows: 1) Bridge Lion at (100, 50), 2) Bridge Tiger at (100, 150)

However, you need to follow the below constraints:

Grounded units (all units but Aviation) are only able to cross the river at either 1) Bridge Lion at (100, 50), 2) Bridge Tiger at (100, 150). To cross, they move use attack_move_unit to first move to the coordinates of either 1) Bridge Lion at (100, 50), 2) Bridge Tiger at (100, 150) and then continue to either attack_move_unit or engage_target_unit or stand_location.


I need to generate one military course of action, where each unit can perform multiple of the above 3 commands, to accomplish the following mission objective while staying within the mentioned constraints:

Move friendly forces from the west side of the river to the east via multiple bridges, destroy all hostile forces, and ultimately seize objective OBJ Lion
East at the top right of the map (coordinates x: 175, y: 175).

The available Friendly and Hostile forces with their respective identification tags, types, and position are defined in the following JSON object:

```
{locations}
```
Remember to respond in JSON format only, do not add any natural language in your response, and only create COAs for friendly troops.

"""

human_instruction_template = """{instruction}You may take maximum of {max_steps} steps.
Here are some examples:"""

HUMAN_INSTRUCTION = HumanMessagePromptTemplate.from_template(human_instruction_template)

human_instruction_reflection_template = """Here are some examples:"""
HUMAN_REFLECTION_INSTRUCTION = HumanMessagePromptTemplate.from_template(human_instruction_reflection_template)

SYSTEM_CRITIQUE_EXISTING_RULES_INSTRUCTION = """"""
SYSTEM_CRITIQUE_ALL_SUCCESS_EXISTING_RULES_INSTRUCTION = """"""
SYSTEM_REFLECTION_INSTRUCTION = """"""

def LLM_PARSER(llm_output, step: int, ai_message: bool) -> Tuple[ChatMessage, str, Dict[str, Any]]:
    pattern = r'(?i)action\s*(?:\d+|)\s*(?::|)\s*'
    action_pattern = r'(?i)\w+\[[^\]]+(?:\]|)'

    match = re.match(pattern, llm_output)
    if match:
        action = llm_output[match.end():]
        content = f"Action {step}: {action}"

        if len(re.findall(action_pattern, action)) > 1:
            return (
                AIMessage(content=content) if ai_message else HumanMessage(content=content),
                'action',
                {'action': ''} # triggers invalid action
            )

        return (
            AIMessage(content=content) if ai_message else HumanMessage(content=content),
            'action',
            {'action': action}
        )

    actions = re.findall(action_pattern, llm_output)
    if len(actions) == 1:
        action = actions[0]
        if action[-1] != ']':
            action += ']'
        content = f"Action {step}: {action}"
        return (
            AIMessage(content=content) if ai_message else HumanMessage(content=content),
            'action',
            {'action': action}
        )
    
    if len(actions) > 1:
        content = re.sub(r"(?i)action\s*(?:\d*|)\s*(?::|)", "", llm_output)
        return (
            AIMessage(content=f"Action {step}: {content}"),
            'action',
            {'action': ''} # triggers invalid action
        )

    # everthing else will be assumed to be a inner thought
    thought_pattern = r'(?i)thought\s*(?:\d+|)\s*(?::|)\s*(.*)'
    match = re.match(thought_pattern, llm_output)
    if match:
        # Extract the thought word and content
        thought_word = match.group(1)
        content = f"Thought {step}: {thought_word.rstrip(':')}"
    else:
        content = f"Thought {step}: {llm_output.rstrip(':')}"
    return (
        AIMessage(content=content) if ai_message else HumanMessage(content=content),
        'thought',
        {}
    )

def OBSERVATION_FORMATTER(observation: str, step: int, *args, **kwargs) -> Tuple[ChatMessage, str]:
    return HumanMessage(content=f"Observation {step}: " + observation.rstrip(':')), 'append'

def STEP_IDENTIFIER(line: str) -> str:
    line = line.strip()
    pattern = re.compile(r'^(?i)action(?:\s+(\d+))?:')
    match = pattern.match(line)
    if match:
        return 'action'
    pattern = re.compile(r'^(?i)observation(?:\s+(\d+))?:')
    match = pattern.match(line)
    if match:
        return 'observation'
    return 'thought'

def CYCLER(lines: str) -> List[str]:
    new_lines = []
    scratch_pad = ''
    for line in lines.split('\n'):

        # line is action
        pattern = re.compile(r'^(?i)action(?:\s+(\d+))?:')
        match = pattern.match(line)
        if match:
            if scratch_pad != '':
                new_lines.append(scratch_pad.strip())
                scratch_pad = ''
            new_lines.append(line)
            continue

        # line is thought
        pattern = re.compile(r'^(?i)thought(?:\s+(\d+))?:')
        match = pattern.match(line)
        if match:
            if scratch_pad != '':
                new_lines.append(scratch_pad.strip())
                scratch_pad = ''
            new_lines.append(line)
            continue

        # step is observation
        scratch_pad += line + '\n'

    # the rest of the scratch pad
    if scratch_pad != '':
        new_lines.append(scratch_pad.strip())
    return new_lines

REFLECTION_PREFIX = '\nReflection:'
def PREVIOUS_TRIALS_FORMATTER(reflections: List[str], include_prefix: bool = True) -> str:
    if reflections == []:
        return ''
    if include_prefix:
        memory_prefix = "You have attempted to solve the task before but failed. The following reflection(s) give a plan to avoid failing the task in the same way you did previously. Use them to improve your strategy of solving the task successfully."
    else:
        memory_prefix = ''
    memory_prefix += '\nReflections:'
    for reflection in reflections:
        memory_prefix += f"\n- {reflection.strip()}"
    return memory_prefix

def STEP_STRIPPER(step: str, step_type: str):
    if step_type == 'observation':
        return re.sub(r'^(?i)observation(?:\s+(\d+))?:', 'Observation:', step)
    if step_type == 'action':
        return re.sub(r'^(?i)action(?:\s+(\d+))?:', 'Action:', step)
    if step_type == 'thought':
        return re.sub(r'^(?i)thought(?:\s+(\d+))?:', 'Thought:', step)
    return step
