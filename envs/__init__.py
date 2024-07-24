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

COA_TASK = """Provide a series of function calls to each friendly unit under your command via natural language, then upon completing your plan, call the finish(JSON_PLAN) function, where JSON_PLAN is the JSON representation of your plan. You are allowed to assign multiple commands to each friendly unit, but must assign at least one to each unit.

1) attack_move_unit(unit_id, target_x, target_y): commands friendly unit to move to target (x, y) coordinate in the map engaging hostile units in its path.
2) engage_target_unit(unit_id, target_unit_id): commands friendly unit to engage with hostile target unit, which is located at the target (x, y) coordinate in the map. If out of range, friendly unit will move to the target unit location before engaging.
3) stand_location(unit_id): commands friendly unit to stand ground at current location and engage any hostile units that are in range.
4) finish(json_plan): call this function upon assigning a command to all friendly units

However, you need to follow the below constraints:

Grounded units (all units but Aviation) are only able to cross the river at either 1) Bridge Lion at (100, 50), 2) Bridge Tiger at (100, 150). To cross, they move use attack_move_unit to first move to the coordinates of either Bridge Lion at (100, 50) or Bridge Tiger at (100, 150) and then continue to either attack_move_unit or engage_target_unit or stand_location.

Remember, it is of vital importance that all friendly units are given commands. The JSON will be provided in the following prompt below.
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
    travelplanner=lambda cfg: [
        {
        'task': f'{cfg.benchmark.task_prefix}{row["query"]}',
        'env_kwargs': {
            'query': row['query']
        },
        'env_name': 'travelplanner',
    } for _, row in joblib.load(cfg.benchmark.task_file).reset_index(drop=True).iterrows()],
)

ENVS = dict(coa=COAEnv, hotpotqa=QAEnv, fever=FeverEnv, alfworld=AlfworldEnv, webshop=WebshopEnv)
