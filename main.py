import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import arXiv_search, format_response, wiki_search

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
    tools = [web_search_tool, wiki_search, arXiv_search]
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
    graph = graph_builder()
    messages = [HumanMessage(content=test_question)]
    response = graph.invoke({"messages": messages})
    answer = response["messages"][-1].content
    formatted_answer = format_response(answer)
    print(f"Agent response: {formatted_answer}")
