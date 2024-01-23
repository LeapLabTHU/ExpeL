from abc import ABC, abstractmethod
from typing import Dict, Any
import gym

class BaseEnv(gym.Env):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action: str, *args, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def success_fn(self) -> bool:
        pass
    
    def is_terminated(self) -> bool:
        return self.terminated

    def is_truncated(self) -> bool:
        return self.curr_step > self.max_steps
