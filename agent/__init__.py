from .base import BaseAgent
from .react import ReactAgent
from .reflect import ReflectAgent
from .expel import ExpelAgent


AGENT = dict(reflection=ReflectAgent, react=ReactAgent, expel=ExpelAgent)
