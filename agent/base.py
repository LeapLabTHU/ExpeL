from abc import ABC, abstractmethod
import re

from prompts.templates.human import human_task_message_prompt


class BaseAgent(ABC):
    """
    Base agent class.
    """
    @abstractmethod
    def run(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def prompt_agent(self) -> str:
        pass

    @abstractmethod
    def _build_agent_prompt(self) -> str:
        pass

    @abstractmethod
    def after_step(self, *args, **kwargs) -> None:
        pass

    def is_terminated(self) -> bool:
        return self.env.is_terminated()

    def is_truncated(self) -> bool:
        return self.env.is_truncated() or (self.token_counter(self.log_history(include_all=True)) > 15800)

    @abstractmethod
    def reset(self, *args, **kwargs) -> None:
        pass

    def log_history(self, include_task: bool = True, include_all: bool = False) -> str:
        all_history = '\n'.join([prompt.content for prompt in self.prompt_history])
        if include_all:
            return all_history

        # only log the task prompt and the agent's response
        reflection_pattern = r'{}'.format(self.format_reflections(self.reflections, include_prefix=False))
        match = re.search(re.escape(reflection_pattern), all_history)
        if not match or match.group() == '' or not include_task:
            task_text_list = human_task_message_prompt.format_messages(task=self.remove_task_suffix(self.task))[0].content.split('\n')
            task_text = '\n'.join(task_text_list)
            pattern = r'{}'.format(re.escape(task_text.strip()) + '.*')
            match = re.search(pattern, all_history)
        if include_task:
            return match.group().lstrip("Now it's your turn!\n") + match.string[match.end():]
        return match.string[match.end():].strip()

    def remove_task_suffix(self, task: str) -> str:
        if self.benchmark_name == 'alfworld':
            return task.split('___')[0]
        return task