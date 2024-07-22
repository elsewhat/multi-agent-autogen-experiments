# Lesson 1: Multi-Agent Conversation and Stand-up Comedy
#From https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen/lesson/2/multi-agent-conversation-and-stand-up-comedy
import os
import autogen

import pprint



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
from autogen import ConversableAgent

cathy = ConversableAgent(
    name="cathy",
    system_message=
    "Your name is Cathy and you are a stand-up comedian focusing on puns.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

joe = ConversableAgent(
    name="joe",
    system_message=
    "Your name is Joe and you are a stand-up comedian with focus on dad jokes with jaw dropping puns"
    "Start the next joke from the punchline of the previous joke.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

chat_result = joe.initiate_chat(
    recipient=cathy, 
    message="I'm Joe. Cathy, let's perform a standup show together",
    max_turns=4,
    summary_method="reflection_with_llm",
    summary_prompt="Summarize the standup show as an article in the local newspaper",    
)

pprint.pprint(chat_result.cost)
pprint.pprint(chat_result.summary)
#pprint.pprint(chat_result.chat_history)