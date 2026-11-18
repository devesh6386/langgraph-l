
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Literal, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv

import operator

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# Testing backend independently
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# Uncomment below only when testing backend directly
# response = chatbot.invoke(
#     {'messages': [HumanMessage(content='Hi my name is ghazni')]},
#     config=CONFIG
# )
# print(chatbot.get_state(config=CONFIG).values)


# Streaming test
'''
for message_chunk, metadata in chatbot.stream(
    {'messages': [HumanMessage(content='What is the recipe for dosa?')]},
    config={'configurable': {'thread_id': 'thread-1'}},
    stream_mode='messages'
):
    if message_chunk.content:
        print(message_chunk.content, end=" ", flush=True)
'''
