
import queue
import uuid
import streamlit as st

from langgraph_mcp_backend import (
    chatbot,
    retrieve_all_threads,
    submit_async_task,
    run_async
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage
)


# ==========================================
# THREAD MANAGEMENT
# ==========================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


async def _load_conversation(thread_id):

    state = await chatbot.aget_state(
        config={
            "configurable": {
                "thread_id": str(thread_id)
            }
        }
    )

    return state.values.get("messages", [])


def load_conversation(thread_id):
    return run_async(_load_conversation(thread_id))


# ==========================================
# SESSION INITIALIZATION
# ==========================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(
        str(thread_id)[:8],
        key=f"thread_{thread_id}",
        use_container_width=True
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                role = "user"

            elif isinstance(msg, AIMessage):
                role = "assistant"

            else:
                continue

            if isinstance(msg.content, str):
                content = msg.content
            else:
                content = str(msg.content)

            temp_messages.append({
                "role": role,
                "content": content
            })

        st.session_state["message_history"] = temp_messages

        st.rerun()


# ==========================================
# MAIN CHAT UI
# ==========================================

st.title("LangGraph MCP Chatbot")

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask me anything...")


# ==========================================
# CHAT EXECUTION
# ==========================================

if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    with st.chat_message("assistant"):

        event_queue = queue.Queue()
        status_holder = {"box": None}

        async def run_stream():

            try:

                async for message_chunk, metadata in chatbot.astream(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    },
                    config=CONFIG,
                    stream_mode="messages"
                ):

                    event_queue.put(("message", message_chunk, metadata))

            except Exception as exc:

                event_queue.put(("error", exc, None))

            finally:

                event_queue.put(("done", None, None))


        future = submit_async_task(run_stream())

        def ai_only_stream():

            while True:

                event_type, data, metadata = event_queue.get()

                if event_type == "done":
                    break

                if event_type == "error":
                    raise data

                message_chunk = data

                if isinstance(message_chunk, ToolMessage):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "MCP Tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"Using tool: {tool_name}",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"Using tool: {tool_name}",
                            state="running"
                        )

                if isinstance(message_chunk, AIMessage):

                    if isinstance(message_chunk.content, str):
                        yield message_chunk.content

        try:

            ai_message = st.write_stream(ai_only_stream())

            if status_holder["box"] is not None:

                status_holder["box"].update(
                    label="Tool execution completed",
                    state="complete",
                    expanded=False
                )

        except Exception as e:

            st.error(f"Chatbot error: {e}")
            ai_message = ""

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })
