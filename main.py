from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


def graph_builder():
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    search_tool = DuckDuckGoSearchRun()
    chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = [search_tool]
    chat_with_tools = chat.bind_tools(tools=tools)

    with open("prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    system_msg = SystemMessage(content=system_prompt)

    def assistant(state: AgentState):
        messages = [system_msg] + state["messages"]
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


# test
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    graph = graph_builder()
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
