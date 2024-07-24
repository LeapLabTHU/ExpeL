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

The mission is taking place in the following map/terrain:

The map is split in two major portions (west and east sides) by a river that runs from north to south right in the middle of a 200 x 200 map. There are two bridges in order to cross the river. Bridges names and their coordinates are as follows: 1) Bridge Lion at (100, 50), 2) Bridge Tiger at (100, 150)
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
