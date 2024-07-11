import getpass
import hydra
from omegaconf import DictConfig
from pathlib import Path
from functools import partial
import os
import random

from agent import AGENT
from langchain.chat_models import ChatOpenAI
from prompts.templates.system import system_message_prompt
from prompts.templates.human import HUMAN_CRITIQUES
from prompts import (
    SYSTEM_INSTRUCTION,
    HUMAN_INSTRUCTION,
    FEWSHOTS,
    REFLECTION_FEWSHOTS,
    HUMAN_REFLECTION_INSTRUCTION,
    SYSTEM_REFLECTION_INSTRUCTION,
    SYSTEM_CRITIQUE_INSTRUCTION,
    RULE_TEMPLATE,
    LLM_PARSER,
    OBSERVATION_FORMATTER,
    STEP_IDENTIFIER,
    CYCLER,
    STEP_CYCLER,
    REFLECTION_PREFIX,
    PREVIOUS_TRIALS_FORMATTER,
    STEP_STRIPPER,
    CRITIQUE_SUMMARY_SUFFIX,
)
from envs import ENVS, INIT_TASKS_FN
from memory import (
    EMBEDDERS,
    RETRIEVERS,
)
from models import LLM_CLS
from utils import save_trajectories_log, load_trajectories_log, shuffled_chunks, get_split_eval_idx_list
from agent.reflect import Count

from dotenv import load_dotenv
load_dotenv()


@hydra.main(version_base=None, config_path="configs", config_name="insight_extraction")
def main(cfg : DictConfig) -> None:

    # Determine if we are either running validation or testing
    if cfg.testing:
        openai_api_key = 'NO_KEY_FOR_TESTING'
    else:
        openai_api_key = os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else getpass.getpass("Enter or paste your OpenAI API Key: ")
    
    # Save the insights based on the log_dir and benchmark inputs
    LOG_PATH = Path('/'.join([cfg.log_dir, cfg.benchmark.name, cfg.agent_type]))
    SAVE_PATH = LOG_PATH / 'extracted_insights'
    SAVE_PATH.mkdir(exist_ok=True)
    
    # Overwriting confirmation
    if not cfg.resume and os.path.exists(f"{SAVE_PATH}/{cfg.run_name}.pkl") and cfg.run_name != 'test':
        while True:
            res = input(f"Are you sure to overwrite '{cfg.run_name}'? (Y/N)\n").lower()
            if res == 'n':
                exit(0)
            elif res == 'y':
                break
    if cfg.resume and cfg.resume_fold < 0:
        print('Specify a fold to resume when resuming a run! (resume_fold=X)')
        exit(1)
    out = load_trajectories_log(
            SAVE_PATH / f"fold_{cfg.resume_fold}" if cfg.resume_fold > -1 else LOG_PATH, 
            run_name=cfg.load_run_name,
            load_log=cfg.resume,
            load_true_log=False
        )
    dicts = out['dicts']
    log = out['log'] if cfg.resume else ''

    cfg.folded = True
    react_agent = AGENT[cfg.agent_type](
        name=cfg.ai_name,
        system_instruction=SYSTEM_INSTRUCTION[cfg.benchmark.name],
        human_instruction=HUMAN_INSTRUCTION[cfg.benchmark.name],
        tasks=INIT_TASKS_FN[cfg.benchmark.name](cfg),
        fewshots=FEWSHOTS[cfg.benchmark.name],
        system_prompt=system_message_prompt,
        env=ENVS[cfg.benchmark.name],
        max_steps=cfg.benchmark.max_steps,
        openai_api_key=openai_api_key,
        llm=cfg.agent.llm,
        llm_builder=LLM_CLS,
        reflection_fewshots=REFLECTION_FEWSHOTS[cfg.benchmark.name],
        reflection_task_prompt=HUMAN_REFLECTION_INSTRUCTION[cfg.benchmark.name],
        reflection_system_instruction=SYSTEM_REFLECTION_INSTRUCTION[cfg.benchmark.name],
        reflection_system_prompt=SYSTEM_INSTRUCTION[cfg.benchmark.name],
        max_relfection_depth=cfg.agent.max_reflection_depth if 'max_reflection_depth' in cfg.agent.keys() else 0,
        system_critique_instructions=SYSTEM_CRITIQUE_INSTRUCTION[cfg.benchmark.name],
        human_critiques=HUMAN_CRITIQUES,
        max_num_rules=cfg.agent.max_num_rules if 'max_num_rules' in cfg.agent.keys() else 0,
        rule_template=RULE_TEMPLATE[cfg.benchmark.name],
        truncate_strategy=cfg.agent.truncate_strategy if 'truncate_strategy' in cfg.agent.keys() else None,
        llm_parser=LLM_PARSER[cfg.benchmark.name],
        observation_formatter=OBSERVATION_FORMATTER[cfg.benchmark.name],
        embedder=EMBEDDERS(cfg.agent.retrieval_kwargs.embedder_type),
        embedder_path=cfg.agent.retrieval_kwargs.embedder_path,
        step_stripper=STEP_STRIPPER[cfg.benchmark.name],
        retriever_cls=RETRIEVERS(cfg.agent.retrieval_kwargs.retriever_type),
        message_splitter=CYCLER[cfg.benchmark.name],
        identifier=STEP_IDENTIFIER[cfg.benchmark.name],
        message_step_splitter=partial(STEP_CYCLER, benchmark=cfg.benchmark.name),
        reflection_prefix=REFLECTION_PREFIX[cfg.benchmark.name],
        previous_trials_formatter=PREVIOUS_TRIALS_FORMATTER[cfg.benchmark.name],
        success_critique_num=cfg.agent.success_critique_num,
        fewshot_strategy=cfg.agent.fewshot_strategy,
        benchmark_name=cfg.benchmark.name,
        reranker=cfg.agent.retrieval_kwargs.reranker,
        buffer_retrieve_ratio=cfg.agent.retrieval_kwargs.buffer_retrieve_ratio,
        critique_truncate_strategy=cfg.agent.critique_truncate_strategy,
        critique_summary_suffix=CRITIQUE_SUMMARY_SUFFIX,
        testing=cfg.testing,
        max_fewshot_tokens = cfg.agent.retrieval_kwargs.max_fewshot_tokens,
    )

    print(f'Loading agent from {LOG_PATH}')
    no_load_list = ['ai_message', 'message_type_format', 'max_num_rules', 'testing', 'human_critiques', 'system_critique_instructions', 'fewshot_strategy', 'success', 'halted', 'fail', 'task_idx', 'prompt_history', 'critique_truncate_strategy', 'success_critique_num', 'reflection_fewshots', 'reflection_system_prompt', 'reflection_prefix', 'reflection_prompt_history', 'reflections', 'previous_trial', 'perform_reflection', 'increment_task', 'reflection_system_kwargs', 'prepend_human_instruction', 'name', 'tasks', 'human_instruction_kwargs', 'all_system_instruction', 'all_fewshots', 'max_steps', 'ordered_summary', 'fewshots', 'system_instruction', 'num_fewshots', 'curr_step', 'log_idx', 'pretask_idx', 'reflect_interaction_idx', 'truncated', 'reward', 'terminated', 'autoqregressive_model_instruction', 'failed_training_task_idx', '_train', 'task', 'eval_idx_list', 'starting_fold', 'starting_idx', 'critique_summary_suffix']
    react_agent.load_checkpoint(dicts[-1], no_load_list=no_load_list)

    random.seed(cfg.seed)
    num_training_tasks = len(INIT_TASKS_FN[cfg.benchmark.name](cfg))
    if not cfg.resume:
        resume = False
    else:
        resume = 'eval_idx_list' in dicts[-1] 
    eval_idx_list = dicts[-1].get('eval_idx_list', get_split_eval_idx_list(dicts[-1], cfg.benchmark.eval_configs.k_folds))

    print(f'eval_idx_list: {eval_idx_list}')
    starting_fold = dicts[-1]['starting_fold'] = dicts[-1].get('critique_summary_fold', 0)

    resume_starting_fold = starting_fold
    critique_summary_log = dicts[-1].get('critique_summary_log', '')

    for k, eval_idxs in enumerate(eval_idx_list):
        if k < starting_fold:
            continue
        training_ids = set(range(num_training_tasks)) - set(eval_idxs)
        (SAVE_PATH / f"fold_{k}").mkdir(exist_ok=True)
        log += f'################## FOLD {k} ##################\n'
        log += react_agent.create_rules(
            list(training_ids),
            cache_fold=k,
            logging_dir=str(SAVE_PATH / f"fold_{k}"),
            run_name=cfg.run_name,
            loaded_dict=dicts[-1] if resume and resume_starting_fold == starting_fold else None,
            loaded_log=critique_summary_log if resume and resume_starting_fold == starting_fold else None,
            eval_idx_list=eval_idx_list,
            saving_dict=True,
        )
        starting_fold += 1

    save_dict = {k: v for k, v in react_agent.__dict__.items() if type(v) in [list, set, str, bool, int, dict, Count] and k not in ['openai_api_key', 'llm']}
    if cfg.folded:
        save_dict['eval_idx_list'] = eval_idx_list
    dicts.append(save_dict)
    save_trajectories_log(
        path=SAVE_PATH, 
        log=log, 
        dicts=dicts,
        run_name=f'{cfg.run_name}',
        save_true_log=False
    )

if __name__ == "__main__":
    main()
