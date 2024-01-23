from typing import List, Callable, Dict, Optional
from copy import deepcopy

class Trajectory:
    def __init__(
        self,
        task: str,
        trajectory: str,
        splitter: Callable,
        identifier: Callable,
        step_splitter: Callable,
        embedder: Optional[Callable] = None,
        reflections: List[str] = None
    ):
        self._task = task
        self._trajectory = trajectory
        self._reflections = deepcopy(reflections)
        self._observations, self._actions, self._thoughts = [], [], []
        for line in splitter(self._trajectory):
            setattr(self, f'_{identifier(line)}s', getattr(self, f'_{identifier(line)}s') + [line])
        self._steps = step_splitter(lines=trajectory, cycler=splitter, step_identifier=identifier)
        self._keys = {'thought': [], 'step': []}
        if embedder is not None:
            self._keys['task'] = [embedder(self.task)]
            for step in self.steps:
                self._keys['step'].append(embedder(step))
            for thought in self.thoughts:
                self._keys['thought'].append(embedder(thought))

    @property
    def task(self) -> str:
        return self._task

    @property
    def steps(self) -> List[str]:
        return self._steps

    @property
    def trajectory(self) -> str:
        return self._trajectory

    @property
    def num_steps(self) -> int:
        return max(len(self.thoughts), len(self.actions), len(self.observations))

    @property
    def observations(self) -> List[str]:
        return self._observations

    @property
    def actions(self) -> List[str]:
        return self._actions

    @property
    def thoughts(self) -> List[str]:
        return self._thoughts

    @property
    def reflections(self) -> List[str]:
        return self._reflections

    @property
    def keys(self) -> Dict[str, List[float]]:
        return self._keys

    def _replace(self, attr, value):
        self.__setattr__(self, attr, value)