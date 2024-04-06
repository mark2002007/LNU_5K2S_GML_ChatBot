from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from llm import llm
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from tools.vector import kg_qa
from tools.cypher import cypher_qa

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
    ),
    Tool.from_function(
        name="Cypher QA",
        description="Provide information about tweets questions using Cypher",
        func = cypher_qa,
        return_direct=True
    ),
    Tool.from_function(
        name="Vector Search Index",
        description="Provides information about tweets using Vector Search",
        func = kg_qa,
        return_direct=True
    )
]

agent_prompt = PromptTemplate.from_template("""
You are a tweet recommendation expert providing suggestions for relevant tweets based on user input. 
Be as helpful as possible and return the most relevant tweets. 
Do not answer any questions that do not relate to tweets or the topics mentioned in the tweets. 
Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
If you have been asked what tools do you have access to, you MUST provide the list of tools available to you.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """    
    print("PROMPT!!!!!!!!!!!!!!!!!:", prompt)
    response = agent_executor.invoke({"input": prompt})
    return response['output']