from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode,tools_condition
from langchain.types import interrupt, Command
from dotenv import load_dotenv

import requests
load_dotenv()
llm=ChatGroq(model='')

@tool
def get_stock_price(symbol:str)->dict:
    """Fetch latest stock price for a given symbol (e.g.'AAPL','TSLA') using 
    Alpha Vantage with API key in the url.
    """
    url=(
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r=requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol:str,quantity:int)->dict:
    """Simulate purchasing a given quantity of a stock symbol.
    HUMAN_IN_THE_LOOP:
    Before confirming the purchase ,this tool will interrupt and 
    wait for a human decision("yes"/anything else.
    """
    decision=interrupt(f"Approve buying{quantity} shares of {symbol}?(yes/no)")
    if isinstance(decision,str)and decision.lower()=="yes":
        return {
            "status":"success",
            "message":f"Purchase order placed for {quantity} share of {symbol}.",
            "symbol":symbol,
            "quantity":quantity,
        }
    else:
        return {
            "status":"cancelled",
            "message":f"Purchase of {quantity}shares of {symbol} was declined by human.",
            "symbol":symbol,
            "quantity":quantity,

        }
tools=[get_stock_price,purchase_stock]
llm_with_tools=llm.bind_tools(tools)

#state
class ChatState(TypedDict):
    message:Annotated[list[BaseMessage],add_messages]

def  
