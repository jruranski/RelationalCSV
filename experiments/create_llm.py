import glob
import pprint

from typing import Any, Iterator, List
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os


def create_vision_llm(model_name="gpt-4o", temperature=0.1, safety_settings=None):
    vision_llm = ChatOpenAI(model=model_name)
    return vision_llm

def create_llm(model_name="gpt-4o-mini", max_output_tokens=16128, temperature=0.1):
   
    llm=ChatOpenAI( 
           model=model_name,
           temperature=temperature,
    )
    return llm
