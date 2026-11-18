
import asyncio
import threading
import os
import requests
import aiosqlite

from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool, BaseTool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# ==========================================
# 1. ASYNC EVENT LOOP
# ==========================================

ASYNC_LOOP = asyncio.new_event_loop()

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

_ASYNC_THREAD = threading.Thread(
    target=start_loop,
    args=(ASYNC_LOOP,),
    daemon=True
)
_ASYNC_THREAD.start()


def submit_async_task(coro):
    return asyncio.run_coroutine_threadsafe(coro, ASYNC_LOOP)


def run_async(coro):
    return submit_async_task(coro).result()


# ==========================================
# 2. LLM
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ==========================================
# 3. TOOLS
# ==========================================

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given stock symbol."""

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {"error": "Alpha Vantage API key missing"}

    url = "https://www.alphavantage.co/query"

    response = requests.get(
        url,
        params={
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key
        },
        timeout=15
    )

    response.raise_for_status()
    return response.json()


# ==========================================
# 4. MCP CLIENT
# ==========================================

client = MultiServerMCPClient({
    "expense": {
        "transport": "streamable_http",
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
    }
})


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print("MCP tools loading error:", e)
        return []


mcp_tools = load_mcp_tools()

tools = [
    search_tool,
    get_stock_price,
    *mcp_tools
]

llm_with_tools = llm.bind_tools(tools)


# ==========================================
# 5. STATE
# ==========================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================
# 6. NODES
# ==========================================

async def chat_node(state: ChatState):

    response = await llm_with_tools.ainvoke(
        state["messages"]
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)


# ==========================================
# 7. CHECKPOINTER
# ==========================================

async def _init_checkpointer():

    conn = await aiosqlite.connect("chatbot2.db")

    saver = AsyncSqliteSaver(conn)

    await saver.setup()

    return saver


checkpointer = run_async(_init_checkpointer())


# ==========================================
# 8. GRAPH
# ==========================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ==========================================
# 9. THREAD MANAGEMENT
# ==========================================

async def _alist_threads():

    all_threads = set()

    async for checkpoint in checkpointer.alist(None):

        thread_id = checkpoint.config["configurable"]["thread_id"]

        all_threads.add(thread_id)

    return list(all_threads)


def retrieve_all_threads():

    return run_async(_alist_threads())
