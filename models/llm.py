import getpass
import json
import os
import requests
import time
from typing import Any, Callable, Dict, List
import yaml

from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import openai


class GPTWrapper:
    def __init__(self, llm_name: str, openai_api_key: str, long_ver: bool):
        self.model_name = llm_name
        if long_ver:
            llm_name = 'gpt-4o-mini'
        self.llm = ChatOpenAI(
            model=llm_name,
            temperature=0.0,
            openai_api_key=openai_api_key,
        )

    def __call__(self, messages: List[ChatMessage], stop: List[str] = [], replace_newline: bool = True) -> str:
        kwargs = {}
        if stop != []:
            kwargs['stop'] = stop
        for i in range(6):
            try:
                output = self.llm(messages, **kwargs).content.strip('\n').strip()
                break
            except openai.error.RateLimitError:
                print(f'\nRetrying {i}...')
                time.sleep(1)
        else:
            raise RuntimeError('Failed to generate response')

        if replace_newline:
            output = output.replace('\n', '')
        return output


# Llama3 class to interact with the Llama model API
class Llama3:
    def __init__(self, llama_url: str, model: str, stream: bool, output: str, messages: List[Dict[str, Any]]):
        self.llama_url = llama_url
        self.model = model
        self.stream = stream
        self.output = output
        self.messages = messages

    def add_message(self, role: str, content: str):
        """Add a message to the list of messages to be sent to the Llama model."""
        if role not in ['system', 'user', 'assistant']:
            raise ValueError("Invalid role")
        self.messages.append({"role": role, "content": content})

    def send_query(self) -> Dict[str, Any]:
        """Send the query to the Llama model and return the response."""
        request = {
            "model": self.model,
            "messages": self.messages[:],
            "stream": self.stream
        }
        # request["messages"][-1]['content'] =  + request["messages"][-1]['content']
        response = requests.post(self.llama_url, json=request)
        response_json = response.json()
        with open(self.output, 'w', encoding='utf-8') as f:
            json.dump(response_json, f, ensure_ascii=False, indent=4)

        self.add_message("assistant", response_json['message']['content'])
        # print(response_json['message']['content'])
        return response_json

# Load the YAML file
with open('../configs/benchmark/coa.yaml', 'r') as file:
    coa_config = yaml.safe_load(file)

def LLM_CLS(llm_name: str, openai_api_key: str, long_ver: bool) -> Callable:
    if 'gpt' in llm_name:
        return GPTWrapper(llm_name, openai_api_key, long_ver)
    elif 'llama' in llm_name:
        return Llama3(llama_url=coa_config['llama_url'], model=coa_config['react_llm_name'],
                      stream=coa_config['stream'], output=os.path.join("logs/coa/expel", "llama_output.json"),
                      messages=[])
    else:
        raise ValueError(f"Unknown LLM model name: {llm_name}")
