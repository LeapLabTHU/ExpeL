import joblib
import json
import random
import json

from .base import BaseEnv
from .coa.coa import COAEnv
from .hotpotqa.hotpotqa import QAEnv
from .fever.fever import FeverEnv
from .alfworld.alfworld import AlfworldEnv
from .webshop.webshop import WebshopEnv
from utils import get_env_name_from_gamefile

# Taken from ReAct Github
idxs = list(range(7405))
random.Random(233).shuffle(idxs)

COA_TASK = """You are a military commander assistant. Your users are military commanders and your role is to help them develop a military courses of action (COA). The military commanders will inform you the mission objective, terrain information, and available friendly and hostile assets before you start developing the COA. Given this information, you will develop a number of courses of action (as specified by the commander) so they can iterate on them with you and pick their favorite one. For each COA to be complete, every friendly unit needs to be assigned one command from the list below. Remember, Hostile units cannot be assigned any command! 

Assign exactly one function call to each friendly unit under your command via natural language, with the available function commands shown below:

1) attack_move_unit(unit_id: int, target_x: int, target_y: int): commands friendly unit to move to target (x, y) coordinate in the map engaging hostile units in its path.
2) engage_target_unit(unit_id: int, target_unit_id: int): commands friendly unit to engage with hostile target unit, which is located at the target (x, y) coordinate in the map. If out of range, friendly unit will move to the target unit location before engaging.
3) stand_location(unit_id: int): commands friendly unit to stand ground at current location and engage any hostile units that are in range.
4) finish(json_plan): call this function upon assigning a command to all friendly units.

Given these functions:
For your first action call attack_move_unit(1, 50, 50).
For your second action, call finish({"unit_id": 1, "action": "attack_move_unit(1, 50, 50)"}).

"""

INIT_TASKS_FN = dict(
    coa=lambda cfg: [
        {
        'task': f"{COA_TASK}: {row['supporting_information']}",
        'env_kwargs': {
            'supporting_information': row['supporting_information'],
        },
        'env_name': 'coa',
    } for _, row in joblib.load(cfg.benchmark.task_file).reset_index(drop=True).iterrows()],
    hotpotqa=lambda cfg: [
        {
        'task': f'{cfg.benchmark.task_prefix}{row["question"]}',
        'env_kwargs': {
            'question': row['question'],
            'key': row['answer'],
        },
        'env_name': 'hotpotqa',
    } for _, row in joblib.load(cfg.benchmark.task_file).reset_index(drop=True).iterrows()],
    # 100 tasks for fever
    fever=lambda cfg: [{
        'task': cfg.benchmark.task_prefix + FeverEnv(idx).reset().replace('Claim: ', ''),
        'env_kwargs': {
            'idx': idx,
        },
        'env_name': 'fever',
    } for idx in idxs[:100]],
    alfworld=lambda cfg: [
        {
        'task': f'{cfg.benchmark.task_prefix}{row["goal"]}',
        'env_kwargs': {
            'config': cfg.benchmark,
            "gamefile": row["gamefile"],
        },
        'env_name': get_env_name_from_gamefile(row['gamefile'])
        } for row in json.load(open(cfg.benchmark.task_file, "r"))
    ],
    webshop=lambda cfg: [
        {
        'task': f'{cfg.benchmark.task_prefix}{row["task"]}',
        'env_kwargs': {
            'session_idx': row["session_idx"],
        },
        'env_name': 'webshop'
        } for row in json.load(open(cfg.benchmark.task_file, "r"))
    ],
)

ENVS = dict(coa=COAEnv, hotpotqa=QAEnv, fever=FeverEnv, alfworld=AlfworldEnv, webshop=WebshopEnv)
