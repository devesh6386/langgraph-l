from langgraph.graph import StateGraph, START, END  
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Literal, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
import sqlite3
load_dotenv()
llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages=state['messages']
    response=llm.invoke(messages)
    return {"messages":[response]}
conn=sqlite3.connect('chatbot.db',check_same_thread=False)#automarically create check same thread=true then code error because multiple thread use after 
#sqlite uses same thread explicitly we are saying then
#connection obejct

#old messages in thread id we can extract

#checkpointer
checkpointer=SqliteSaver(conn=conn)
graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot=graph.compile(checkpointer=checkpointer)
def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        all_threads.add(thread_id)

    return list(all_threads)
