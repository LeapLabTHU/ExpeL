from langchain.prompts.chat import HumanMessagePromptTemplate


system_template = """You are {ai_name}. {instruction}"""

system_message_prompt = HumanMessagePromptTemplate.from_template(system_template)
