import random
from typing import List, Dict, Callable, Union, Any, Tuple
import re
from functools import partial

from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, Document
import numpy as np
from openai.error import InvalidRequestError
from scipy.spatial.distance import cosine

from agent import ReflectAgent, ReactAgent
from agent.reflect import Count
from utils import random_divide_list, save_trajectories_log, get_env_name_from_task
from memory import Trajectory

from copy import deepcopy

class ExpelAgent(ReflectAgent):
    def __init__(self,
                 system_critique_instructions: Dict[str, str],
                 human_critiques: Dict[str, PromptTemplate],
                 rule_template: PromptTemplate,
                 max_num_rules: Union[int, str],
                 truncate_strategy: str,
                 embedder: Callable,
                 embedder_path: str,
                 step_stripper: Callable,
                 retriever_cls: Callable,
                 success_critique_num: int,
                 fewshot_strategy: str,
                 critique_truncate_strategy: str,
                 benchmark_name: str,
                 critique_summary_suffix: str,
                 max_fewshot_tokens: int,
                 reranker: str,
                 buffer_retrieve_ratio: int,
                 *args,
                 **kwargs,
                 ) -> None:
        self.benchmark_name = benchmark_name
        self.system_critique_instructions = system_critique_instructions
        self.human_critiques = human_critiques
        self.max_num_rules = max_num_rules
        self.rule_template = rule_template
        self.truncate_strategy = truncate_strategy
        self.critique_truncate_strategy = critique_truncate_strategy
        self.embedder = embedder(model_name=embedder_path)
        self.fewshot_strategy = fewshot_strategy
        self.retriever_cls = retriever_cls
        self.step_stripper = step_stripper
        self.success_critique_num = success_critique_num
        self.reranker = reranker
        self.buffer_retrieve_ratio = buffer_retrieve_ratio
        self.failed_training_task_idx = []
        self.critique_summary_suffix = critique_summary_suffix
        self.max_fewshot_tokens = max_fewshot_tokens
        self.eval_successes = []
        self.succeeded_trial_history: Dict[str, Trajectory] = {}
        self.failed_trial_history: Dict[str, Trajectory] = {}
        self.critiques = {}
        self.all_success_critiques = {}
        self.past_reflections = {}
        self.rule_items = []
        self.rule_items_with_count = []
        self.cache_rules = {}
        self._train = True
        super().__init__(benchmark_name=benchmark_name, *args, **kwargs)
        self.idx2task = {idx: task['task'] for idx, task in enumerate(self.tasks)}
        self.task2idx = {task['task']: idx for idx, task in enumerate(self.tasks)}

    @property
    def training(self) -> bool:
        return self._train

    def train(self) -> None:
        self._train = True

    def eval(self) -> None:
        self._train = False

    def next_task(self) -> bool:
        # storing reflections
        if self.task not in self.past_reflections:
            self.past_reflections[self.task] = []
        if self.reflections != []:
            self.past_reflections[self.task].append(self.reflections[-1])

        # only reflect on the task if the task is training task
        if self.training:
            # record the tasks
            history = self.log_history(include_task=False)
            trajectory = Trajectory(
                task=self.remove_task_suffix(self.task),
                trajectory=history,
                reflections=self.reflections,
                splitter=self.message_splitter,
                identifier=self.identifier,
                step_splitter=self.message_step_splitter,
            )
            self.succeeded_trial_history = deepcopy(self.succeeded_trial_history)
            self.failed_trial_history = deepcopy(self.failed_trial_history)

            # first time doing the task
            if self.task not in self.failed_trial_history:
                self.succeeded_trial_history[self.task] = []
                self.failed_trial_history[self.task] = []
            # if changing task, reflect accordingly
            if self.increment_task:
                if self.is_success():
                    self.succeeded_trial_history[self.task].append(trajectory)
                else:
                    self.failed_trial_history[self.task].append(trajectory)
                    # record the task index that failed
                    self.failed_training_task_idx.append(self.task_idx)
            else:
                self.failed_trial_history[self.task].append(trajectory)
        
        return ReflectAgent.next_task(self)

    ################# CRITIQUES #################

    def task_critique(self, task: str, return_log: bool = False) -> Union[None, str]:
        # only critique if the task has success
        if task not in self.critiques:
            self.critiques[task] = []
        if return_log:
            log = ''
        if self.succeeded_trial_history[task] != []:
            # if first time critiquing the task
            for traj in self.succeeded_trial_history[task]:
                success_history = traj.trajectory.strip()
                # forming critiques by comparing successful and failed trials
                for fail_history in self.failed_trial_history[task]:
                    critiques: str = self.prompt_critique(
                        success_history=success_history,
                        fail_history=fail_history.trajectory.lstrip(),
                    )
                    if return_log:
                        log += success_history + '\n' + fail_history.trajectory.strip() + '\n' + critiques + '\n\n'
                    critiques: List[str] = critiques.split('\n- ' if not self.testing else '\\n- ')
                    self.critiques[task].extend(critiques)
                pattern = r"\s*\([^()]*\)"
                self.critiques[task] = [re.sub(pattern, '', critique).strip().strip('- ') for critique in self.critiques[task]]
                # removing empty critique
                self.critiques[task] = [critique for critique in self.critiques[task] if critique != '']

        if return_log:
            return log

    def success_critique(self, training_ids: List[int]) -> None:
        # make sure to only take the training ids, assuming theres only one success trajectory per task
        all_success = []
        for task in self.succeeded_trial_history:
            idx = self.task2idx[task]
            if idx in training_ids and len(self.succeeded_trial_history[task]) > 0:
                all_success.append((self.remove_task_suffix(task), self.succeeded_trial_history[task][0].trajectory))
        all_success = random_divide_list(all_success, self.success_critique_num)
        # refresh the success critiques
        self.all_success_critiques = {}
        for success_chunk in all_success:
            success_trials = '\n\n'.join([task + '\n' + trajectory for task, trajectory in success_chunk])
            critiques: str = self.prompt_critique(success_history=success_trials.strip(), fail_history=None)
            critiques: List[str] = critiques.split('\n- ' if not self.testing else '\\n- ')
            key = '\n'.join([task for task, _ in success_chunk])
            self.all_success_critiques[key] = critiques
            pattern = r"\s*\([^()]*\)"
            self.all_success_critiques[key] = [re.sub(pattern, '', critique).strip().strip('- ') for critique in self.all_success_critiques[key]]
            # removing empty critique
            self.all_success_critiques[key] = [critique for critique in self.all_success_critiques[key] if critique != '']

    def failure_critique(self) -> None:
        self.all_fail_critiques = {}
        for task, failed_trajectories in self.failed_trial_history.items():
            # only critiquing if the task has failed more than once
            if len(failed_trajectories) > 1:
                failed_trials = '\n\n'.join([traj.trajectory for traj in failed_trajectories])
                if self.token_counter(failed_trials) > 13000:
                    print('TRUNCATING FAILED TRIALS')
                    if self.critique_truncate_strategy == 'random':
                        idx = np.random.choice(range(len(failed_trajectories)), size=(len(failed_trajectories) - 1,))
                        failed_trials = '\n\n'.join([traj.trajectory for i, traj in enumerate(failed_trajectories) if i in idx])
                    elif self.critique_truncate_strategy == 'longest':
                        filtered_idx = max(range(len(failed_trajectories)), key=lambda i: self.token_counter(failed_trajectories[i].trajectory))
                        failed_trials = '\n\n'.join([traj.trajectory for i, traj in enumerate(failed_trajectories) if i != filtered_idx])
                    elif self.critique_truncate_strategy == 'shortest':
                        filtered_idx = min(range(len(failed_trajectories)), key=lambda i: self.token_counter(failed_trajectories[i].trajectory))
                        failed_trials = '\n\n'.join([traj.trajectory for i, traj in enumerate(failed_trajectories) if i != filtered_idx])
                    else:
                        raise NotImplementedError
                critiques: str = self.prompt_critique(success_history=None, fail_history=failed_trials.strip(), task=task)
                critiques: List[str] = critiques.split('\n- ' if not self.testing else '\\n- ')
                self.all_fail_critiques[task] = critiques
                pattern = r"\s*\([^()]*\)"
                self.all_fail_critiques[task] = [re.sub(pattern, '', critique).strip().strip('- ') for critique in self.all_fail_critiques[task]]
                # removing empty critique
                self.all_fail_critiques[task] = [critique for critique in self.all_fail_critiques[task] if critique != '']
            else:
                self.all_fail_critiques[task] = []

    def _build_critique_prompt(self, success_history: Trajectory, fail_history: str = Trajectory, existing_rules: List[str] = None, task: str = None, reflections: List[str] = None) -> List[HumanMessage]:
        critique_history = []
        if reflections is not None:
            critique_type = 'all_reflection'
        elif fail_history is not None and success_history is not None:
            critique_type = 'compare'
        elif fail_history is None and success_history is not None:
            critique_type = 'all_success'
        elif fail_history is not None and success_history is None:
            critique_type = 'all_fail'
        if existing_rules is not None:
            critique_type += '_existing_rules'
        if existing_rules == []:
            existing_rules = ['']

        # system prompt
        critique_history.extend(self.system_prompt.format_messages(
            instruction=self.system_critique_instructions[critique_type].format(
                fewshots=[],
            ),
            ai_name='an advanced reasoning agent that can critique past task trajectories of youself' if existing_rules is None \
                else 'an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories',
        ))
        # task_prompt
        human_format_dict = dict(instruction='',)
        if critique_type == 'compare':
            human_format_dict['task'] = task
        if fail_history is not None:
            human_format_dict['fail_history'] = fail_history
            human_format_dict['task'] = task
        if success_history is not None:
            human_format_dict['success_history'] = success_history
        if reflections is not None:
            human_format_dict['reflections_list'] = '- ' + '\n- '.join(reflections)
        if existing_rules is not None:
            human_format_dict['existing_rules'] = '\n'.join([f'{i}. {r}' for i, r in enumerate(existing_rules, 1)])
        human_critique_summary_message = self.human_critiques[critique_type].format_messages(**human_format_dict)[0]
        critique_summary_suffix = self.critique_summary_suffix['full'] if self.max_num_rules <= len(self.rule_items_with_count) else self.critique_summary_suffix['not_full']
        human_critique_summary_message.content = human_critique_summary_message.content + critique_summary_suffix
        critique_history.append(human_critique_summary_message)
        return critique_history

    def prepare_new_eval(self) -> None:
        self.succeeded_trial_history = {}
        self.failed_trial_history = {}

    def prompt_critique(
        self, success_history: Trajectory, fail_history: Trajectory,
        existing_rules: List[str] = None, task: str = None, reflections: List[str] = None) -> str:
        critique_history = self.collapse_prompts(
            self._build_critique_prompt(success_history, fail_history, existing_rules, task if task is None else self.remove_task_suffix(task), reflections)
        )
        print("\n###################################\n")
        if self.testing:
            print('###################################')
            for prompt in critique_history:
                self.print_message(prompt, self.token_counter)
            return input()
        # just use the base llm for critiques
        try:
            returns = self.llm(critique_history, replace_newline=False)
        except InvalidRequestError:
            returns = self.long_context_llm(critique_history, replace_newline=False)
        for i, m in enumerate(critique_history):
            self.print_message(m)
            if i == len(critique_history) - 1:
                print(returns)
        return returns

    ################# EVALUATION #################

    def run(self, mode: str, eval_idx: int = None, reset: bool = True):
        # normal training step
        if mode == 'train':
            return ReflectAgent.run(self, reset)
        # testing step
        if mode == 'eval':
            self.task = self.tasks[eval_idx]['task']
            self.set_env(self.tasks[eval_idx]['env_kwargs'], max_steps=self.max_steps)
            ret = ReactAgent.run(self, reset)
            if self.is_success():
                self.eval_successes.append(eval_idx)
            return ret
        raise NotImplementedError

    def create_rules(
        self,
        training_ids: List[int],
        cache_fold: int = None,
        load_cache_fold: int = None,
        logging_dir: str = None,
        run_name: str = 'run',
        loaded_dict: Dict[str, Any] = None,
        loaded_log: str = None,
        eval_idx_list: List[int] = None,
        saving_dict: bool = False,
    ) -> str:
        if load_cache_fold is not None:
            self.rules = '\n'.join([f'{i}. {item}' for i, item in enumerate(self.cache_rules.get(load_cache_fold, []), 1)])
            return

        def extend_rules(rule_items: List[str], success_history: str = None, fail_history: str = None, task: str = None, reflections: List[str] = None) -> List[str]:
            llm_output: str = self.prompt_critique(
                success_history=success_history,
                fail_history=fail_history,
                existing_rules=rule_items,
                reflections=reflections,
                task=task,
            )
            parsed_operations = parse_rules(llm_output)

            # update the rule_items with counter
            self.rule_items_with_count = update_rules(self.rule_items_with_count, parsed_operations, list_full = self.max_num_rules+5 <= len(self.rule_items_with_count))

            new_ordered_rules_str = [rule[0] for rule in self.rule_items_with_count]
            return new_ordered_rules_str, llm_output

        # Shuffling the rules into a pool
        resume_flag = fail_resume_flag = loaded_dict is None
        if resume_flag:
            self.rule_items = []
            self.rule_items_with_count: List[tuple(str, int)] = []
        agent_dicts = []
        if loaded_log is None:
            all_logs = '################ Compare Critiques ################\n'
        else:
            all_logs = loaded_log
        for training_id in training_ids:
            training_task = self.idx2task[training_id]
            if (loaded_dict is not None and loaded_dict['critique_summary_section'] == 'compare' and \
                loaded_dict['critique_summary_idx'][0] == training_id):
                resume_flag = True
                # if there are still failed tasks to do, then dont continue, otherwise do the next idx's critiques
                if len(self.failed_trial_history[training_task]) - 1 <= loaded_dict['critique_summary_idx'][1]:
                    fail_resume_flag = True
                    continue
            elif not resume_flag:
                continue
            if self.succeeded_trial_history[training_task] != []:
                # if first time critiquing the task
                for traj in self.succeeded_trial_history[training_task]:
                    success_history = traj.trajectory.strip()
                    # forming critiques by comparing successful and failed trials
                    for e, fail_history in enumerate(self.failed_trial_history[training_task]):
                        if fail_resume_flag:
                            pass
                        elif e <= loaded_dict['critique_summary_idx'][1]:
                            continue
                        fail_resume_flag = True
                        self.rule_items, llm_output = extend_rules(self.rule_items, success_history, fail_history.trajectory.strip(), training_task)
                        all_logs += training_task + '\n' + success_history + '\n' + fail_history.trajectory.strip() + f'\n-------\n{llm_output}\n-------\n' +'\n- ' + '\n- '.join([str(r) + " {" + str(c) + "}" for r, c in self.rule_items_with_count]) + '\n\n'
                        if saving_dict:
                            save_dict = {k: deepcopy(v) for k, v in self.__dict__.items() if type(v) in [list, set, str, bool, int, dict, Count] and k not in ['openai_api_key', 'llm']}
                            save_dict['critique_summary_section'] = 'compare'
                            save_dict['critique_summary_idx'] = (training_id, e)
                            save_dict['critique_summary_fold'] = cache_fold if cache_fold is not None else 0
                            save_dict['critique_summary_log'] = all_logs
                            save_dict['eval_idx_list'] = eval_idx_list
                            agent_dicts.append(save_dict)
                            save_trajectories_log(path=logging_dir, log=all_logs, dicts=agent_dicts, run_name=run_name, save_true_log=False)

        # SUCCESS
        if loaded_log is None or loaded_dict['critique_summary_section'] in ['compare']:
            all_logs += '\n\n################ SUCCESS CRITIQUES ################\n'
        else:
            all_logs = loaded_log
        all_success = []
        if loaded_dict is None or loaded_dict['critique_summary_section'] == 'compare':
            for training_id in training_ids:
                all_success = []
                for idx, task in enumerate(self.succeeded_trial_history):
                    if idx in training_ids and len(self.succeeded_trial_history[task]) > 0:
                        all_success.append((task, self.succeeded_trial_history[task][0].trajectory))
                all_success = random_divide_list(all_success, self.success_critique_num)
        else:
            all_success = loaded_dict['critique_summary_all_success']
        for success_chunk in all_success:
            if (loaded_dict is not None and loaded_dict['critique_summary_section'] == 'success' and \
                loaded_dict['critique_summary_idx'] == success_chunk):
                resume_flag = True
                continue
            elif not resume_flag:
                continue
            success_trials = '\n\n'.join([self.remove_task_suffix(task) + '\n' + trajectory for task, trajectory in success_chunk])
            self.rule_items, llm_output = extend_rules(self.rule_items, success_trials.strip(), None)
            all_logs += success_trials.strip() + f'\n-------\n{llm_output}\n-------' + '\n- ' + '\n- '.join([str(r) + " {" + str(c) + "}" for r, c in self.rule_items_with_count]) + '\n\n'
            if saving_dict:
                save_dict = {k: deepcopy(v) for k, v in self.__dict__.items() if type(v) in [list, set, str, bool, int, dict, Count] and k not in ['openai_api_key', 'llm']}
                save_dict['critique_summary_all_success'] = all_success
                save_dict['critique_summary_idx'] = success_chunk
                save_dict['critique_summary_section'] = 'success'
                save_dict['critique_summary_fold'] = cache_fold if cache_fold is not None else 0
                save_dict['critique_summary_log'] = all_logs
                save_dict['eval_idx_list'] = eval_idx_list
                agent_dicts.append(save_dict)
                save_trajectories_log(path=logging_dir, log=all_logs, dicts=agent_dicts, run_name=run_name, save_true_log=False)

        # numbered list format
        self.rules = '\n'.join([f"{i}. {item}" for i, item in enumerate(self.rule_items, 1)])
        if cache_fold is not None:
            self.cache_rules[cache_fold] = list(self.rule_items)
        return all_logs

    def insert_before_task_prompt(self):
        # if training then reflect
        if self.training:
            return ReflectAgent.insert_before_task_prompt(self)
        # if eval, add the manual
        if not self.no_rules:
            self.prompt_history.append(
                self.rule_template.format_messages(rules=self.rules)[0]
            )

    def insert_after_task_prompt(self):
        pass

    def after_step(self) -> None:
        pass

    def setup_vectorstore(self) -> None:
        self.keys2task = {'thought': {}, 'task': {}, 'step': {}, 'reflection': {}, 'action': {}}
        self.docs = []
        combined_history = dict(self.succeeded_trial_history)
        if isinstance(self.all_fewshots, list):
            for fewshot in self.all_fewshots:
                if self.benchmark_name in ['coa', 'hotpotqa', 'fever']:
                    task = fewshot.split('\n')[0]
                    trajectory = '\n'.join(fewshot.split('\n')[1:])
                elif self.benchmark_name == 'webshop':
                    task = '\n'.join(fewshot.split('\n')[:2])
                    trajectory = '\n'.join(fewshot.split('\n')[2:])
                cleaned_traj = Trajectory(
                    task=self.remove_task_suffix(task),
                    trajectory=trajectory,
                    reflections=[],
                    splitter=self.message_splitter,
                    identifier=self.identifier,
                    step_splitter=partial(
                        self.message_step_splitter,
                        stripper=self.step_stripper
                    ),
                )
                combined_history.update({task: [cleaned_traj]})
        elif isinstance(self.all_fewshots, dict):
            fewshot_offset = 100000
            for env_task, fewshots in self.all_fewshots.items():
                for fewshot in fewshots:
                    if self.benchmark_name in ['alfworld']:
                        task = '\n'.join(fewshot.split('\n')[:3]) + '___' + str(fewshot_offset)
                        trajectory = '\n'.join(fewshot.split('\n')[3:])
                    cleaned_traj = Trajectory(
                        task=self.remove_task_suffix(task),
                        trajectory=trajectory,
                        reflections=[],
                        splitter=self.message_splitter,
                        identifier=self.identifier,
                        step_splitter=partial(
                            self.message_step_splitter,
                            stripper=self.step_stripper
                        ),
                    )
                    combined_history.update({task: [cleaned_traj]})
                    fewshot_offset += 1
        for task in combined_history:
            if combined_history[task] != []:
                self.docs.append(Document(page_content=self.remove_task_suffix(task), metadata={'type': 'task', 'task': task, 'env_name': get_env_name_from_task(task, self.benchmark_name)}))
            for i, traj in enumerate(combined_history[task]):
                cleaned_traj = Trajectory(
                    task=self.remove_task_suffix(task),
                    trajectory=traj.trajectory,
                    reflections=list(traj.reflections),
                    splitter=self.message_splitter,
                    identifier=self.identifier,
                    step_splitter=partial(
                        self.message_step_splitter,
                        stripper=self.step_stripper
                    ),
                )
                cleaned_thoughts: List[str] = cleaned_traj.thoughts
                cleaned_steps: List[str] = cleaned_traj.steps
                cleaned_reflections: List[str] = cleaned_traj.reflections
                cleaned_actions: List[str] = cleaned_traj.actions
                self.docs.extend([Document(page_content=action, metadata={'type': 'action', 'task': task, 'env_name': get_env_name_from_task(task, self.benchmark_name)}) for action in cleaned_actions])
                self.docs.extend([Document(page_content=thought, metadata={'type': 'thought', 'task': task, 'env_name': get_env_name_from_task(task, self.benchmark_name)}) for thought in cleaned_thoughts])
                self.docs.extend([Document(page_content=step, metadata={'type': 'step', 'task': task, 'env_name': get_env_name_from_task(task, self.benchmark_name)}) for step in cleaned_steps])
                if cleaned_reflections != []:
                    self.docs.extend([Document(page_content=reflection, metadata={'type': 'reflection', 'task': task, 'env_name': get_env_name_from_task(task, self.benchmark_name)}) for reflection in cleaned_reflections])
                for thought in cleaned_thoughts:
                    self.keys2task['thought'][thought] = (task, i)
                for step in cleaned_steps:
                    self.keys2task['step'][step] = (task, i)
                for reflection in cleaned_reflections:
                    self.keys2task['reflection'][reflection] = (task, i)
                for action in cleaned_actions:
                    self.keys2task['action'][action] = (task, i)
        self.combined_history = combined_history

    def update_dynamic_prompt_components(self, reset:bool = False):
        if reset:
            ReactAgent.update_dynamic_prompt_components(self)
            return
        # do not dynamically update during training
        if self.training or self.fewshot_strategy == 'none':
            return
        old_fewshots = '\n\n'.join(self.fewshots)

        def filtered_vectorstore(fewshot_strategy: str, docs: List[Document]):
            strat2filter = {
                'task_similarity': 'task', 'step_similarity': 'step',
                'reflection_similarity': 'reflection', 'thought_similarity': 'thought',
                'action_similarity': 'action'
            }
            if fewshot_strategy == 'random':
                subset_docs = list(filter(lambda doc: doc.metadata['type'] == strat2filter['task_similarity'] and doc.metadata['env_name'] == self.env.env_name, docs))
            else:
                subset_docs = list(filter(lambda doc: doc.metadata['type'] == strat2filter[fewshot_strategy] and doc.metadata['env_name'] == self.env.env_name, docs))
            # adhoc filtering for webshop
            if self.benchmark_name == 'webshop':
                filtered_subset_docs = []
                for doc in subset_docs:
                    trajectory = self.combined_history[doc.metadata['task']][0].trajectory
                    if 'Observation: Invalid action!' not in trajectory and \
                        'think[]' not in trajectory and \
                            len(trajectory.split('Observation: You have clicked'))>=3:
                        filtered_subset_docs.append(doc)
            else:
                filtered_subset_docs = subset_docs

            return FAISS.from_documents(filtered_subset_docs, self.embedder)

        def topk_docs(queries: Dict[str, str], query_type: str):
            # retrieve enough fewshots, filtering the ones that are too long
            fewshot_docs = self.vectorstore.similarity_search(queries[query_type], k=self.num_fewshots*self.buffer_retrieve_ratio)
            if self.fewshot_strategy == 'random':
                random.shuffle(fewshot_docs)
            fewshots = []
            current_tasks = set()
            def fewshot_doc_token_count(fewshot_doc):
                return self.token_counter(self.combined_history[fewshot_doc.metadata['task']][0].trajectory)
            # default no reranker if thought is empty
            if self.reranker == 'none' or (self.reranker == 'thought' and queries['thought'] == ''):
                fewshot_docs = list(fewshot_docs)
            elif self.reranker == 'len':
                fewshot_docs = list(sorted(fewshot_docs, key=fewshot_doc_token_count, reverse=True))
            elif self.reranker == 'thought' and queries['thought'] != '':
                fewshot_tasks = set([doc.metadata['task'] for doc in fewshot_docs])
                subset_docs = list(filter(lambda doc: doc.metadata['type'] == 'thought' and doc.metadata['env_name'] == self.env.env_name and doc.metadata['task'] in fewshot_tasks, list(self.docs)))
                fewshot_docs = sorted(subset_docs, key=lambda doc: cosine(self.embedder.embed_query(doc.page_content), self.embedder.embed_query(queries['thought'])))
            elif self.reranker == 'task':
                fewshot_tasks = set([doc.metadata['task'] for doc in fewshot_docs])
                subset_docs = list(filter(lambda doc: doc.metadata['type'] == 'thought' and doc.metadata['env_name'] == self.env.env_name and doc.metadata['task'] in fewshot_tasks, list(self.docs)))
                fewshot_docs = sorted(subset_docs, key=lambda doc: cosine(self.embedder.embed_query(doc.page_content), self.embedder.embed_query(queries['task'])))
            else:
                raise NotImplementedError
            for fewshot_doc in fewshot_docs:
                idx, shortest_fewshot = sorted(enumerate([traj.trajectory for traj in self.combined_history[fewshot_doc.metadata['task']]]), key=lambda x: len(x[1]))[0]

                # if fewshot is using more than 1k tokens OR
                # if the fewshot is the same as the current task OR
                # if the fewshot is the same as one of the current fewshots, skip it
                if self.token_counter(shortest_fewshot) > self.max_fewshot_tokens or \
                    self.task == fewshot_doc.metadata['task'] or fewshot_doc.metadata['task'] in current_tasks:
                    continue
                fewshots.append(self.combined_history[fewshot_doc.metadata['task']][idx].task + '\n' + shortest_fewshot)

                current_tasks.add(fewshot_doc.metadata['task'])
                if len(fewshots) == self.num_fewshots:
                    break

            return fewshots

        self.setup_vectorstore()
        self.vectorstore = filtered_vectorstore(self.fewshot_strategy if self.fewshot_strategy not in ['rotation', 'task_thought_similarity'] else 'task_similarity', docs=list(self.docs))

        if self.prompt_history == []:
            queries = {'task': self.step_stripper(self.remove_task_suffix(self.task), step_type='task')}
        else:
            history = self.log_history(include_task=False)
            # used to index
            trajectory = Trajectory(
                task=self.remove_task_suffix(self.task),
                trajectory=history,
                reflections=list(self.reflections),
                splitter=self.message_splitter,
                identifier=self.identifier,
                step_splitter=self.message_step_splitter,
            )
            steps = self.message_splitter(trajectory.steps[-1])
            step_types = [self.identifier(step) for step in steps]
            if 'observation' not in step_types and self.fewshot_strategy == 'step': # if the step is not complete, use the previous step
                steps = self.message_splitter(trajectory.steps[-2])
                step_types = [self.identifier(step) for step in steps]
            cleaned_step = '\n'.join([self.step_stripper(step, step_type) for step, step_type in zip(steps, step_types)])
            queries = {
                'task': self.step_stripper(self.remove_task_suffix(self.task), step_type='task'),
                'thought': '' if len(trajectory.thoughts) < 1 or trajectory.thoughts[0] == '' else self.step_stripper(trajectory.thoughts[-1], step_type='thought'),
                'step': cleaned_step,
                'action': self.step_stripper(trajectory.actions[-1], step_type='action') if len(trajectory.actions) > 1 else '',
            }

        if self.fewshot_strategy == 'random':
            self.vectorstore = filtered_vectorstore('random', docs=list(self.docs))
            self.fewshots = topk_docs(queries=queries, query_type='task')
        elif self.fewshot_strategy == 'rotation':
            last_step_type = self.identifier(self.message_splitter(trajectory.trajectory)[-1])
            # use task to retrieve
            if self.prompt_history == [] or len(trajectory.thoughts) < 1 or trajectory.thoughts[0] == '':
                self.vectorstore = filtered_vectorstore('task_similarity', docs=list(self.docs))
                self.fewshots = topk_docs(queries=queries, query_type='task')
            else:
                if last_step_type == 'thought':
                    self.vectorstore = filtered_vectorstore('thought_similarity', docs=list(self.docs))
                    self.fewshots = topk_docs(queries=queries, query_type='thought')
                elif last_step_type == 'observation':
                    self.vectorstore = filtered_vectorstore('step_similarity', docs=list(self.docs))
                    self.fewshots = topk_docs(queries=queries, query_type='step')
        elif self.fewshot_strategy == 'task_thought_similarity':
            # use task to retrieve
            if self.prompt_history == [] or len(trajectory.thoughts) < 1 or trajectory.thoughts[0] == '':
                self.vectorstore = filtered_vectorstore('task_similarity', docs=list(self.docs))
                self.fewshots = topk_docs(queries=queries, query_type='task')
            else:
                self.vectorstore = filtered_vectorstore('thought_similarity', docs=list(self.docs))
                self.fewshots = topk_docs(queries=queries, query_type='thought')
        elif self.fewshot_strategy == 'task_similarity':
            # retrieve task as the query, and task as the keys for successful trials
            self.vectorstore = filtered_vectorstore('task_similarity', docs=list(self.docs))
            self.fewshots = topk_docs(queries=queries, query_type='task')
        # both thought and reflection retrieve based on the latest thought
        elif self.fewshot_strategy == 'thought_similarity':
            if self.prompt_history == [] or len(trajectory.thoughts) < 1 or trajectory.thoughts[0] == '':
                ReactAgent.update_dynamic_prompt_components(self)
            else:
                # use the latest thoughts to retrieve fewshots
                self.vectorstore = filtered_vectorstore('thought_similarity', docs=list(self.docs))
                self.fewshots = topk_docs(queries=queries, query_type='thought')
        elif self.fewshot_strategy == 'action_similarity':
            if self.prompt_history == [] or len(trajectory.actions) < 1:
                ReactAgent.update_dynamic_prompt_components(self)
            else:
                # use the latest thoughts to retrieve fewshots
                self.vectorstore = filtered_vectorstore('action_similarity', docs=list(self.docs))
                self.fewshots = topk_docs(queries=queries, query_type='action')
        elif self.fewshot_strategy == 'step_similarity':
            if self.prompt_history == [] or len(trajectory.observations) < 1:
                ReactAgent.update_dynamic_prompt_components(self)
            else:
                self.vectorstore = filtered_vectorstore('step_similarity', docs=list(self.docs))
                self.fewshots = topk_docs(queries=queries, query_type='step')
        else:
            raise NotImplementedError
        # storing the new fewshots and replacing the current ones from prompt_history
        new_fewshots = '\n\n'.join(self.fewshots)
        replaced = False
        for i, history_message in enumerate(self.prompt_history):
            if old_fewshots in history_message.content:
                message_type = type(history_message)
                self.prompt_history[i] = message_type(content=history_message.content.replace(old_fewshots, new_fewshots))
                replaced = True
                break
        if not replaced and self.testing:
            self.prompt_history.append(HumanMessage(content="WARNING. Fewshots haven't been replaced."))

# Utils function
def parse_rules(llm_text):
    pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
    matches = re.findall(pattern, llm_text)

    res = []
    banned_words = ['ADD', 'AGREE', 'EDIT']
    for operation, text in matches:
        text = text.strip()
        if text != '' and not any([w in text for w in banned_words]) and text.endswith('.'):
        # if text is not empty
        # if text doesn't contain banned words (avoid weird formatting cases from llm)
        # if text ends with a period (avoid cut off sentences from llm)
            if 'ADD' in operation:
                res.append(('ADD', text))
            else:
                res.append((operation.strip(), text))
    return(res)

def retrieve_rule_index(rules, operation):
    operation_rule_text = operation[1]
    for i in range(len(rules)):
        if rules[i][0] in operation_rule_text:
            return i

def is_existing_rule(rules, operation_rule_text):
    for i in range(len(rules)):
        if rules[i][0] in operation_rule_text:
            return True
    return False

# Given list of tuples with (rule text, number of edits) and tuple of (operations, text), returns updated list of tuples
def update_rules(rules: List[Tuple[str, int]], operations: List[Tuple[str, str]], list_full: bool = False) -> List[Tuple[str, int]]:
    # remove problematic operations
    delete_indices = []
    for i in range(len(operations)):
        operation, operation_rule_text = operations[i]
        operation_type = operation.split(' ')[0]
        rule_num = int(operation.split(' ')[1]) if ' ' in operation else None

        if operation_type == 'ADD':
            if is_existing_rule(rules, operation_rule_text): # if new rule_text is an existing rule ('in')
                delete_indices.append(i)
        else:
            if operation_type == 'EDIT':
                if is_existing_rule(rules, operation_rule_text): # if rule is matching ('in') existing rule, change it to AGREE 
                    rule_num = retrieve_rule_index(rules, (operation, operation_rule_text))
                    operations[i] = (f'AGREE {rule_num+1}', rules[rule_num][0])
                elif (rule_num is None) or (rule_num > len(rules)):   # if rule doesn't exist, remove
                    delete_indices.append(i)
                    
            elif operation_type == 'REMOVE' or operation_type == 'AGREE':
                if not is_existing_rule(rules, operation_rule_text): # if new operation_rule_text is not an existing rule
                    delete_indices.append(i)

    operations = [operations[i] for i in range(len(operations)) if i not in delete_indices] # remove problematic operations

    for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']: # Order is important
        for i in range(len(operations)):
            operation, operation_rule_text = operations[i]
            operation_type = operation.split(' ')[0]
            if operation_type != op:
                continue

            if operation_type == 'REMOVE': # remove rule: -1
                rule_index = retrieve_rule_index(rules, (operation, operation_rule_text)) # if rule_num doesn't match but text does
                remove_strength = 3 if list_full else 1
                rules[rule_index] = (rules[rule_index][0], rules[rule_index][1]-remove_strength) # -1 (-3 if list full) to the counter
            elif operation_type == 'AGREE': # agree with rule: +1
                rule_index = retrieve_rule_index(rules, (operation, operation_rule_text)) # if rule_num doesn't match but text does
                rules[rule_index] = (rules[rule_index][0], rules[rule_index][1]+1) # +1 to the counter
            elif operation_type == 'EDIT': # edit the rule: +1 // NEED TO BE AFTER REMOVE AND AGREE
                rule_index = int(operation.split(' ')[1])-1
                rules[rule_index] = (operation_rule_text, rules[rule_index][1]+1) # +1 to the counter
            elif operation_type == 'ADD': # add new rule: +2
                rules.append((operation_rule_text, 2))
    rules = [rules[i] for i in range(len(rules)) if rules[i][1] > 0] # remove rules when counter reach 0
    rules.sort(key=lambda x: x[1], reverse=True)

    return rules
