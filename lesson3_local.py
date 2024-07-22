# Lesson 3: Sequential Chats and Customer Onboarding
#From https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen/lesson/4/reflection-and-blogpost-writing
import pprint
import os

#Load .env with OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()

llm_config = {
    "config_list":
    [
        {
            "api_type": "ollama",
            "model": "llama3:8b",
            "client_host": "http://host.docker.internal:11434",
            "seed": 42,
            "price": [0.0,0.0]
        }
    ]
}

#The openai llm_config works
#llm_config = {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"],"price": [0.00015, 0.00060]}


task = '''
        Write a concise but engaging blogpost about
       DeepLearning.AI. Make sure the blogpost is
       within 100 words.
       '''

import autogen

writer = autogen.AssistantAgent(
    name="Writer",
    system_message="You are a writer. You write engaging and concise " 
        "blogpost (with title) on given topics. You must polish your "
        "writing based on the feedback you receive and give a refined "
        "version. Only return your final work without additional comments.",
    llm_config=llm_config,
)

critic = autogen.AssistantAgent(
    name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    system_message="You are a critic. You review the work of "
                "the writer and provide constructive "
                "feedback to help improve the quality of the content.",
)

res = critic.initiate_chat(
    recipient=writer,
    message=task,
    max_turns=5,
    summary_method="last_msg"
)