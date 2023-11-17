# Libary Modules needed for this script: slack_bolt, os, json, llama_index, openai
from pydantic import BaseModel, Field
from pprint import pprint
from typing import Type, Any


# APP related
from app.api.api_v1.models import (
    UserMessage,
    BotMessage,
    WebsiteQuestion,
    AddSource,
    Datasource,
)
from app.api.api_v1.utils import search_pinecone, load_file
from app.api.api_v1.database import (
    retrieve_chat_history,
    insert_chat_history,
    upsert_datasource,
    retrieve_agent_datasources,
)

# LLMs and LangChain
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

# Data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)

# import nest_asyncio

# nest_asyncio.apply()

"""" AGENT TOOLS """

ai_agent_name = "Information Systems"


async def rag(query: str, texts_only: bool = False) -> str | None:
    """Useful for finding information in a previously saved source.  Uses parameter 'query':str which should be the ENTIRE question from the user.
    Returns a string of context from the context source or None. When using the texts_only = Fals parameter, the response will be in JSON and the metadata score that is highest is the most likely answer so favor the text from that result"""

    context = await search_pinecone(query=query, top_k=3, texts_only=texts_only)
    print(">>>>>>>>>>>>>>>>>> HERE IS THE RETURNED RAG CONTEXT <<<<<<<<<<<<<<<<<<<<<<")
    pprint(context)

    return context


def pandas_agent(csv_path: str, question: str) -> Any:
    """Useful for doing anyalysis from a CSV.  Use the question parameter to specify the question for pandas agent"""

    df = pd.read_csv(csv_path)

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.0)
    # Show the intermediate steps because this agent sometimes doesn't put the full answer in the final answer.
    agent = create_pandas_dataframe_agent(llm, df=df, verbose=True, return_intermediate_steps=True)

    return agent(question)



async def add_source(
    ai_agent: str,
    file_type: str,
    file_link: str,
    created_by:str,
    title: str | None = None,
    sparse_summary: str | None = None,
    recursive_scraping: bool | None = True
) -> str | bool:
    """Useful when a user wishes to add a data source. Make sure to create a title and a sparse summary for the user if they don't provide one. If the link goes to an html page or website, you should ask the uer if the want to do 'Recursive Scraping' or just scrape that one page"""
    # metadata for the vectorstore
    meta = []
    ai_agent = ai_agent_name
    if title is not None:
        meta.append({"title": title})
    # load the file in the vectorstore and backup to s3
    s3_key = await load_file(
        file_type=file_type, file_link=file_link, metadata_to_save=meta, recursive_scraping=recursive_scraping
    )
    if s3_key:
        # file loaded, now add to the datasource collection in mongo
        ds = Datasource(
            file_link=file_link,
            file_type=file_type,
            ai_agent=ai_agent,
            created_by=created_by,
            title=title,
            sparse_summary=sparse_summary,
            s3_key=s3_key,
        )
        new_ds = await upsert_datasource(ds)

    return s3_key


async def list_sources() -> list[Datasource] | None:
    """Useful for getting a list of all the currently saved data sources that the you have access to. Use the rag tool to get information within these sources"""
    print("RUNNING LIST SOURCES")
    r = await retrieve_agent_datasources(agent=ai_agent_name)
    pprint(r)
    return r


""" AGENT """


async def agent(payload: UserMessage):
    
    llm = ChatOpenAI(model="gpt-4-1106-preview")

    rag_tool = StructuredTool.from_function(rag)
    rag_tool.coroutine = rag

    add_source_tool = StructuredTool.from_function(add_source)
    add_source_tool.coroutine = add_source
    add_source_tool.args_schema = Datasource

    list_sources_tool = StructuredTool.from_function(list_sources)
    list_sources_tool.coroutine = list_sources

    pandas_agent_tool = StructuredTool.from_function(pandas_agent)
    # pandas_agent_tool.return_direct = True

    tools = [rag_tool, add_source_tool, list_sources_tool, pandas_agent_tool]

    system_prompt = f"""
        You are an ambitious and friendly genius named Salkie, internally your ai_agent name is {ai_agent_name}. You have degrees in Information Systems, Business Administration, Logic, Liberal Arts, Law and Computer Science. You have the amazing ability to read JSON data and make sense of it easily.  You are presently working for a high ranking employee at the Salk Institute who's username is {payload.username}, you help them answer and plan their next action for any questions, challenges or research they are doing.
        If using the rag tool always use the ENTIRE question from the user for the query parameter.
        For any tool you want to use, make sure you have values for all of the tool's requied parameters, otherwise don't use that tool.
        You love to respond in Markdown syntax and always with uniquely cited sources (as clickable links) at the end. You will use as many tools as needed to answer the user's question.  You will be rewarded for taking some extra steps to find ALL the information requested from the user.
    """

    existing_messages = await retrieve_chat_history(
        username=payload.username, botname="slackbot-is"
    )

    # Always add the system message
    messages = [("system", system_prompt)]
    # Make the list of tuples format that LangChain expects
    for m in existing_messages:
        messages.append((m["role"], m["content"]))
    # add the current message
    messages += [
        ("user", payload.message),
        # placeholder for the agent scratchpad too.
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]

    # Make template, add the messages
    prompt = ChatPromptTemplate.from_messages(messages)
    # pprint(f"-------------Here is the current Prompt-----------------/n{prompt}")

    # Bind tools to llm
    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )
    # use an OpenAI Functions compatible Tool Schema and Agent Scratchpad
    agent_schema = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(
            x["intermediate_steps"]
        ),
    }
    # Use LCEL syntax to build the agent
    agent = agent_schema | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Get the agent response object.
    a = await agent_executor.ainvoke({"input": payload.message})

    # pprint(f">>>>>>>>>>Unformatted answer<<<<<<<<<<<<<<<<")
    print(a)

    # append response and user question to history
    add_history_user_message = await insert_chat_history(
        username=payload.username,
        botname="slackbot-is",
        role="user",
        content=payload.message,
    )
    add_history_ai_message = await insert_chat_history(
        username=payload.username,
        botname="slackbot-is",
        role="ai",
        content=a["output"],
    )

    return {
        "username": payload.username,
        "context": payload.context,
        "contextType": payload.contextType,
        "message": a["output"],
        "sources": [],
    }
