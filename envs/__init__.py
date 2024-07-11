import joblib
import json
import random
import json

from .base import BaseEnv
from .hotpotqa.hotpotqa import QAEnv
from .fever.fever import FeverEnv
from .alfworld.alfworld import AlfworldEnv
from .webshop.webshop import WebshopEnv
from utils import get_env_name_from_gamefile

# Taken from ReAct Github
idxs = list(range(7405))
random.Random(233).shuffle(idxs)

INIT_TASKS_FN = dict(
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

ENVS = dict(hotpotqa=QAEnv, fever=FeverEnv, alfworld=AlfworldEnv, webshop=WebshopEnv)
