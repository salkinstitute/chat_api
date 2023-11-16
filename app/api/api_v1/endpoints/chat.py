import os, time, pprint
from pydantic import Field
from fastapi import APIRouter

# APP related
from ..models import (
    UserMessage,
    BotMessage,
    WebsiteQuestion,
)

from ..utils import search_pinecone, load_file
from ..database import retrieve_chat_history, insert_chat_history
from ..agents.information_systems import agent as info_sys_agent

# LLMs and LangChain
from openai import OpenAI
from langchain.chat_models.openai import ChatOpenAI

from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain.chains import LLMMathChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor


# load_dotenv()


router = APIRouter(prefix="/chat", tags=["Chat"])


# Entry point, uses bot name to direct to function
@router.post("/{bot_name}")
async def chat_with_bot(payload: UserMessage, bot_name: str) -> BotMessage:
    match bot_name.lower().strip():
        case "information sytems":
            return await info_sys_agent(payload)
        case _:  # default
            return await info_sys_agent(payload)


#  Manually testing each load type first
@router.get("")
async def root():
    pl = "https://www.cnn.com"
    load_file(
        file_type="website",
        file_link=pl,
        metadata_to_save=[{"source": pl, "test_data": True}],
    )
    return {"message": "FILE LOADED"}


# tests
