
import sys
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage

from utils.agent_tools import list_spaces, hybrid_search

load_dotenv()

# 1. Local LLM
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    num_ctx=8192,
)

# 2. Instructions for the agent
SYSTEM_PROMPT = """
You are a Kubernetes documentation assistant.

For Kubernetes questions:
1. Use hybrid_search to find relevant documentation.
2. Read the retrieved passages.
3. If needed, search again with a better query.
4. Answer using only evidence from retrieved documents.
5. Cite the source URLs from the search results.
6. If there is insufficient evidence, say so.

Do not invent documentation or citations.
"""

# 3. Create the agent
agent = create_agent(
    model=llm,
    tools=[list_spaces, hybrid_search],
    system_prompt=SYSTEM_PROMPT,
)


def main():
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "What is the purpose of kube-proxy?"

    print("Question:", question)

    # 4. Run the LangGraph agent
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 15},
    )

    # 5. Inspect which tools were called
    print("\n--- TOOL CALLS ---")

    for message in result["messages"]:
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print("Tool:", call["name"])
                print("Arguments:", call["args"])

        if isinstance(message, ToolMessage):
            print("Tool result:", str(message.content)[:500])

    # 6. Display final answer
    print("\n--- FINAL ANSWER ---")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
