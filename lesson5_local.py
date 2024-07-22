# Lesson 5: Coding and financial analysis
#From https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen/lesson/6/coding-and-financial-analysis
import pprint
import os


from typing_extensions import Annotated

#Load .env with OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()

llm_config = {
    "config_list":
    [
        {
            "api_type": "ollama",
            "model": "gemma2:27b",
            "client_host": "http://host.docker.internal:11434",
            "seed": 42,
            "price": [0.0,0.0]
        }
    ]
}



#The openai llm_config works
#llm_config = {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"],"price": [0.00015, 0.00060]}

from autogen.coding import LocalCommandLineCodeExecutor
executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
)

from autogen import ConversableAgent, AssistantAgent

code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply=
    "Please continue. If everything is done, reply 'TERMINATE'.",
)

code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

code_writer_agent_system_message = code_writer_agent.system_message

print(code_writer_agent_system_message)

import datetime

today = datetime.datetime.now().date()
message = f"Today is {today}. "\
"Create a plot showing stock gain YTD for NVDA and TLSA. "\
"Make sure the code is in markdown code block and save the figure"\
" to a file ytd_stock_gains.png."""

chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=message,
)