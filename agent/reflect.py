from typing import List, Callable

from openai.error import InvalidRequestError
from langchain.schema import HumanMessage

from agent.react import ReactAgent
from utils import Count
import re

class ReflectAgent(ReactAgent):
    """
    A Generic Reflection Agent.
    """
    def __init__(self,
                 reflection_fewshots: List[str],
                 reflection_task_prompt: Callable,
                 reflection_system_instruction: str,
                 max_relfection_depth: int,
                 message_splitter: Callable,
                 identifier: Callable,
                 message_step_splitter: Callable,
                 reflection_prefix: str,
                 previous_trials_formatter: Callable,
                 *args,
                 **kwargs,
                 ) -> None:
        self.reflection_counter = Count(max_relfection_depth)
        self.reflection_fewshots = reflection_fewshots
        self.reflection_task_prompt = reflection_task_prompt
        self.message_splitter = message_splitter
        self.identifier = identifier
        self.message_step_splitter = message_step_splitter
        self.reflection_prefix = reflection_prefix
        self.format_reflections = previous_trials_formatter
        self.reflection_prompt_history = []
        self.reflections = []
        self.previous_trial = []
        self.formatted_reflection = None
        self.perform_reflection = False
        self.increment_task = False
        ai_name = 'an advanced reasoning agent that can improve based on self refection'
        self.reflection_system_kwargs = dict(instruction=reflection_system_instruction, ai_name=ai_name)
        super().__init__(*args, **kwargs)

    def run(self, reset: bool = True, *args, **kwargs) -> None:
        if self.perform_reflection and not self.is_success():
            self.reflect()
        ReactAgent.run(self, reset)
        if self.reflection_counter.is_maximum() or self.is_success():
            self.increment_task = True


    def step(self) -> None:
        ReactAgent.step(self)
        trial = self.prompt_history[self.history_index].content.split(self.remove_task_suffix(self.task), 1)[-1].strip()
        steps = self.message_step_splitter(
            lines=trial,
            cycler=self.message_splitter,
            step_identifier=self.identifier)
        self.previous_trial.append(HumanMessage(content=steps[-1]))


    def reflect(self) -> None:
        self._format_reflection_scratchpad()
        self.reflection_prompt_history.append(HumanMessage(content=self.reflection_prefix))
        reflection = self.prompt_reflection()
        self.reflections.append(reflection)
        self.formatted_reflection = self.format_reflections(self.reflections)
        print(self.formatted_reflection)
        # wipe the history for a new round
        self.previous_trial = []

    def insert_before_task_prompt(self) -> None:
        if self.formatted_reflection is not None:
            self.prompt_history.append(HumanMessage(content=self.formatted_reflection))

    def prompt_reflection(self) -> str:
        self.reflection_prompt_history = self.collapse_prompts(self.reflection_prompt_history)
        if self.benchmark_name == 'webshop':
            # match the last "Observation:"
            pattern = r"\nObservation: (.*[\n]+)+Next plan:.*"
            matches = re.findall(pattern, self.reflection_prompt_history[-1].content)
            if 'Ran out of steps' in matches[-1]:
                add_text = "\nObservation: Ran out of steps! TASK FAILED\n\nNext plan:\n"
            elif 'Repeated action' in matches[-1]:
                add_text = "\nObservation: Repeated action! TASK FAILED\n\nNext plan:\n"
            else:
                add_text = "\nObservation: Wrong item! TASK FAILED\n\nNext plan:\n"

            new_history = self.reflection_prompt_history[-1].content.split(matches[-1])
            new_history = ''.join(new_history[:-1]) + add_text

            self.reflection_prompt_history[-1].content = new_history

        if self.testing:
            print('###################################')
            for prompt in self.reflection_prompt_history:
                self.print_message(prompt, self.token_counter)
            return input()
        try:
            return self.llm(self.reflection_prompt_history, stop=['\n', '\n\n'])
        except InvalidRequestError:
            return self.long_context_llm(self.reflection_prompt_history, stop=['\n', '\n\n'])

    def _build_reflection_prompt(self) -> None:
        # avoid building reflection prompt if it already exists
        if self.reflection_prompt_history != []:
            return
        system_prompt = self.system_prompt.format_messages(**self.reflection_system_kwargs)
        self.reflection_prompt_history.extend(system_prompt)
        self._build_fewshot_prompt(
            fewshots=self.reflection_fewshots,
            prompt_history=self.reflection_prompt_history,
            instruction_prompt=self.reflection_task_prompt,
            instruction_prompt_kwargs={},
            prompt_type='reflect_type',
        )
        self.reflection_prompt_history.append(HumanMessage(content=f'Previous trial:\n{self.remove_task_suffix(self.task)}'))
        self.reflect_interaction_idx = len(self.reflection_prompt_history)
        for message in self.previous_trial:
            self.reflection_prompt_history.append(message)

    def _format_reflection_scratchpad(self) -> str:
        lines = [ref.content for ref in self.reflection_prompt_history[self.reflect_interaction_idx:]]
        lines_by_tokens = sorted(lines, key=lambda x: self.token_counter(x))
        while self.token_counter(''.join(lines)) > 12000:
            ind = lines.index(lines_by_tokens.pop(-1))
            line = lines[ind]
            lines[ind]  = line.split(':')[0] + ': ...'
        combined_message = HumanMessage(content='\n'.join(lines))
        self.reflection_prompt_history = self.reflection_prompt_history[:self.reflect_interaction_idx]
        self.reflection_prompt_history.append(combined_message)

    def reset(self, *args, **kwargs) -> None:
        ReactAgent.reset(self, *args, **kwargs)
        self.reflection_prompt_history = []
        self._build_reflection_prompt()
        if self.increment_task:
            self.reflections = []
            self.reflection_counter.reset()
            self.formatted_reflection = None
            self.previous_trial = []

    @property
    def history_index(self) -> int:
        return -1

    def next_task(self) -> bool:
        # increment task if reflection counter is at max OR if the agent is successful
        if self.increment_task:
            self.task_idx += 1
            if self.job_not_done():
                self.task = self.tasks[self.task_idx]['task']
                self.set_env(self.tasks[self.task_idx]['env_kwargs'], max_steps=self.max_steps)
                self.perform_reflection = False
                # wipe the history for a new task
                self.previous_trial = []
        # if there are more tasks, perform reflection
        if self.job_not_done() and not self.increment_task:
            self.perform_reflection = True
            self.reflection_counter.increment()
        self.reset()
        self.env.reset()
        if self.increment_task:
            self.increment_task = False
            return True
        return False

    def update_stats(self) -> None:
        # only count when finished trying for this task
        if self.increment_task:
            if not self.is_success() and self.is_truncated():
                self.halted += 1
            else:
                if self.reward:
                    self.success += 1
                else:
                    self.fail += 1
