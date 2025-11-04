import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import arXiv_search, format_response, python_repl_tool, wiki_search
from utils import preprocess_file_for_agent

load_dotenv()

from langchain_tavily import TavilySearch

web_search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    include_raw_content=False,
    include_images=False,
)


def graph_builder():
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    chat = ChatOpenAI(
        model="gpt-4o-mini",  # "gpt-3.5-turbo",
        temperature=0,
    )
    tools = [web_search_tool, wiki_search, arXiv_search, python_repl_tool]
    chat_with_tools = chat.bind_tools(tools=tools)

    with open("prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
    SYSTEM_PROMPT_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)

    def assistant(state: AgentState):
        non_sys_msg = [msg for msg in state["messages"] if msg.type != "system"]
        messages = [SYSTEM_PROMPT_MESSAGE] + non_sys_msg
        print("=== Messages passed to assistant ===")
        for msg in messages:
            print(f"{msg.type}: {msg.content}")
        print("===================================")
        response = chat_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()


if __name__ == "__main__":
    print("\n" + "-" * 30 + " Agent Starting " + "-" * 30)
    # Test the agent with a sample question
    # test_question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    test_question = ".rewsna eht sa 'tfel' drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI"
    test_question = {
        "task_id": "1f975693-876d-457b-a649-393859e79bf3",
        "question": "Hi, I was out sick from my classes on Friday, so I'm trying to figure out what I need to study for my Calculus mid-term next week. My friend from class sent me an audio recording of Professor Willowbrook giving out the recommended reading for the test, but my headphones are broken :(\n\nCould you please listen to the recording for me and tell me the page numbers I'm supposed to go over? I've attached a file called Homework.mp3 that has the recording. Please provide just the page numbers as a comma-delimited list. And please provide the list in ascending order.",
        "Level": "1",
        "file_name": "1f975693-876d-457b-a649-393859e79bf3.mp3",
    }
    test_question = {
        "task_id": "cca530fc-4052-43b2-b130-b30968d8aa44",
        "question": "Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation.",
        "Level": "1",
        "file_name": "cca530fc-4052-43b2-b130-b30968d8aa44.png",
    }
    test_question = {
        "task_id": "f918266a-b3e0-4914-865d-4faa564f1aef",
        "question": "What is the final numeric output from the attached Python code?",
        "Level": "1",
        "file_name": "f918266a-b3e0-4914-865d-4faa564f1aef.py",
    }
    print(f"Task:{test_question['question']}\n\n")
    print("--Processing task and task files (if any) for the agent--")
    task_description = preprocess_file_for_agent(
        task_text=test_question["question"], task_file_name=test_question["file_name"]
    )
    print(f"--Task Description--\n\n{task_description}")
    print("--Building Graph--\n\n")
    graph = graph_builder()
    messages = [HumanMessage(content=task_description)]
    print("--Invoking Graph--\n\n")
    response = graph.invoke({"messages": messages}, config={"recursion_limit": 5})
    answer = response["messages"][-1].content
    print("--Formatting the response--\n\n")
    formatted_answer = format_response(answer)
    print(f"Agent response: {formatted_answer}")
