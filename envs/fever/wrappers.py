import json
import gym
import numpy as np

from utils import normalize_answer
    
DATA_DIR = "data/fever"
HOTPOTQA_SPLIT_FILE = {
  "train": "hotpot_train_v1.1_simplified.json",
  "dev": "hotpot_dev_v1_simplified.json",
  "test": "hotpot_test_v1_simplified.json",
}

FEVER_SPLIT_FILE = {
  "train": "train.jsonl",
  "dev": "paper_dev.jsonl",
}

class FeverWrapper(gym.Wrapper):
  def __init__(self, env, split):
    super().__init__(env)
    
    data_path = f"{DATA_DIR}/{FEVER_SPLIT_FILE[split]}"
    with open(data_path, "r") as json_file:
      json_list = list(json_file)

    data = []
    for json_str in json_list:
      json_str = json.loads(json_str)
      label = json_str["label"]
      claim = json_str["claim"]
      data.append((claim, label))

    self.data = data
    self.data_idx = 0
    self.split = split

  def reset(self, seed=None, return_info=False, options=None, idx=None):
    self.env.reset(seed=seed, return_info=return_info, options=options)
    try:
      self.env.step('')
    except:
      pass
    self.env.reset(seed=seed, return_info=return_info, options=options)
    self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
    observation = f"Claim: {self.data[self.data_idx][0]}"
    info = self._get_info()
    return (observation, info) if return_info else observation

  def _get_info(self):
    return {
      "steps": self.steps, 
      "answer": self.answer,
      "question": self.data[self.data_idx][0], 
      "fever_split": self.split
    }

  def get_reward(self, info):
    if info['answer'] is not None:
      label = normalize_answer(self.data[self.data_idx][1])
      pred = normalize_answer(info['answer'])
      if label == pred:
        return 1
    return 0

  def step(self, action):
    obs, _, done, info = self.env.step(action)
    reward = self.get_reward(info)
    if done:
      obs = f"Episode finished, reward = {reward}\n"
      info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
      info.update({'em': reward, 'reward': reward, 'f1': reward})
    return obs, reward, done, info
    
  def __len__(self):
    return len(self.data)
