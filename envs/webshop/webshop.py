import math
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
import numpy as np

from typing import Any, Dict, List, Tuple
from envs.base import BaseEnv

 
# Type in the URL of the webshop server:
# If local:
WEBSHOP_URL = "http://127.0.0.1:3000"

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )

def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
        f'{WEBSHOP_URL}/item_page/{session}/'
        f'{asin}/{query_string}/{page_num}/{str(options).replace("#","%23")}' # FIXING '#' in url ISSUE
      )
    elif page_type == 'item_sub':
      url = (
        f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{str(options).replace("#","%23")}' # FIXING '#' in url ISSUE
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{str(options).replace("#","%23")}'
      )
    html = requests.get(url).text # type: ignore
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url: # type: ignore
                    processed_t = f'[[{t}]]'
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = round(float(visible_texts[idx + 1]), 2)
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        
        # Prompt harmonization
        observation = observation.replace('\nWebShop ', '')
        observation = observation.replace('\nInstruction: ', '')
        observation = observation.replace('[Search]\n', '[Search]')

        return clean_str(observation), info

class WebshopEnv(BaseEnv):
    def __init__(self, session_idx: str, max_steps: int = 15):
        self.session_idx = session_idx
        self.max_steps = max_steps
        self.env_name = 'webshop'
        self.task = "instruction-following shopping task. The agent interacted with an online store website in order to buy the requested item based on an instruction."

        self.reset()


    def reset(self):
        self.curr_step = 1
        self.reward = False
        self.terminated = False
        self.last_action = None 

        self.session = {'session': self.session_idx, 'page_type': 'init'}
        observation, info = webshop_text(**self.session)
        self.session.update(info)

    def success_fn(self) -> bool:
       return self.reward

    def step(self, action: str) -> Tuple[str, bool, bool, bool, int]:
        done = False
        observation_ = None
        
        try:
            if action == 'reset':
                self.session = {'session': self.session_idx, 'page_type': 'init'}
            elif action.startswith('think['):
                observation = 'OK.'
            elif action.startswith('search['):
                assert self.session['page_type'] == 'init'
                query = action[7:-1]
                self.session = {'session': self.session_idx, 'page_type': 'search',
                                'query_string': query, 'page_num': 1, 'fake_page_num' : 1}
            elif action.startswith('click['): 
                button = action[6:-1]
                if button == 'Buy Now':
                    assert self.session['page_type'] == 'item'
                    self.session['page_type'] = 'end'
                    self.terminated = True
                elif button == 'Back to Search':
                    assert self.session['page_type'] in ['search', 'item_sub', 'item']
                    self.session = {'session': self.session_idx, 'page_type': 'init'}
                elif button == 'Next >':
                    # assert False # ad hoc page limitation
                    assert self.session['page_type'] == 'search'
                    assert self.session['page_num'] < math.ceil(self.session['max_products'] / 10) # Seemed already capped at 50
                    self.session['page_type'] = 'search'
                    self.session['page_num'] += 1

                elif button == '< Prev':
                    assert self.session['page_type'] in ['search', 'item_sub', 'item']
                    if self.session['page_type'] == 'search':
                        assert self.session['page_num'] > 1
                        self.session['page_num'] -= 1
                        self.session['page_type'] = 'search'
                    elif self.session['page_type'] == 'item_sub':
                        self.session['page_type'] = 'item'
                    elif self.session['page_type'] == 'item':
                        self.session['page_type'] = 'search'
                        self.session['options'] = {}
                elif button in ACTION_TO_TEMPLATE:
                    assert self.session['page_type'] == 'item'
                    self.session['page_type'] = 'item_sub'
                    self.session['subpage'] = button
                else:
                    if self.session['page_type'] == 'search':
                        assert button in self.session.get('asins', [])  # must be asins
                        self.session['page_type'] = 'item'
                        self.session['asin'] = button
                    elif self.session['page_type'] == 'item':
                        assert 'option_types' in self.session
                        assert button in self.session['option_types'], (button, self.session['option_types'])  # must be options
                        option_type = self.session['option_types'][button]
                        if not 'options' in self.session:
                            self.session['options'] = {}
                        self.session['options'][option_type] = button
                        observation_ = f'You have clicked {button}.'
            else:
                assert False
        except AssertionError:
            observation_ = 'Invalid action!'
            if invalid_repeat(action=action, last_action=self.last_action):
                self.truncated = True
                self.terminated = True
                observation_ = 'Repeated action!'

        observation, info = webshop_text(**self.session)
        # update the max number of products of a query when we search
        if observation_ not in ['Invalid action!','Repeated action!'] and action.startswith('search['):
            pattern = r'\(Total results: (\d+)\)'
            max_products = int(re.findall(pattern, observation)[0])
            self.session.update({'max_products': max_products})
        if observation_:
            observation = observation_

        # OK from react code for observation:
        if action.startswith('think['):
            observation = 'OK.'

        self.session.update(info)
        reward = info.get('reward', 0.0)

        self.curr_step += 1

        if self.is_truncated() and not self.is_terminated():
            observation += ('\n\n' if observation != '' else '') + 'Ran out of steps! TASK FAILED'

        self.reward = (reward==1)
        self.last_action = action
            
        return observation, self.reward, self.is_terminated(), self.is_truncated(), self.curr_step
        
def invalid_repeat(action: str, last_action: str) -> bool:
    if last_action is None:
        return False
    not_start_list = ['search[', 'think[i apologize', 'think[end', 'think[i\'m sorry', 'think[apolog']
    for word in not_start_list:
        if action.lower().startswith(word) and last_action.lower().startswith(word):
            return True
    return False
