# AI Agents in LangGraph
---

## Build an Agent from Scratch

- Based on a ReAct, reasoning + acting, pattern.
  - The LLM first thinks and decides on the action to take.
  - The action is them executed.
  - Process repeats.

- **Simple ReAct Application by Simon Willison.**
  
  - *Importing dependencies and initilizing the agent.*
  ```
  import openai
  import re
  import httpx
  import os
  from dotenv import load_dotenv

  _ = load_dotenv()
  from openai import OpenAI

  client = OpenAI()

  chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}]
  )
  
  # Testing
  chat_completion.choices[0].message.content
  ```

  - Creating an Agent.
  ```
  class Agent:
    def __init__(self, system=""):
      self.system = system
      self.messages = []
      if self.system:
          self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
      self.messages.append({"role": "user", "content": message})
      result = self.execute()
      self.messages.append({"role": "assistant", "content": result})
      return result

    def execute(self):
      completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=self.messages)
      return completion.choices[0].message.content
  ```

  - Prompt for ReAct Pattern.
  ```
  prompt = """
  You run in a loop of Thought, Action, PAUSE, Observation.
  At the end of the loop you output an Answer
  Use Thought to describe your thoughts about the question you have been asked.
  Use Action to run one of the actions available to you - then return PAUSE.
  Observation will be the result of running those actions.

  Your available actions are:

  calculate:
  e.g. calculate: 4 * 7 / 3
  Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

  average_dog_weight:
  e.g. average_dog_weight: Collie
  returns average weight of a dog when given the breed

  Example session:

  Question: How much does a Bulldog weigh?
  Thought: I should look the dogs weight using average_dog_weight
  Action: average_dog_weight: Bulldog
  PAUSE

  You will be called again with this:

  Observation: A Bulldog weights 51 lbs

  You then output:

  Answer: A bulldog weights 51 lbs
  """.strip()
  ```
  
  - Creating an example calculate function as part of the known actions for the agent:
  ```
  def calculate(what):
    return eval(what)

  def average_dog_weight(name):
    if name in "Scottish Terrier": 
      return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
      return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
      return("a toy poodles average weight is 7 lbs")
    else:
      return("An average dog weights 50 lbs")

  known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
  }
  ```

  - Creating an agent by runnning `abot = Agent(prompt)`

  - The ensuing result of `result = abot("How much does a toy poodle weigh?")` is
  ```
  Thought: I should look up the average weight of a Toy Poodle using the average_dog_weight action.
  Action: average_dog_weight: Toy Poodle
  PAUSE
  ```

  - The result is then checked with the calculator, `result = average_dog_weight("Toy Poodle")` with the output as:
  ```
  'a toy poodles average weight is 7 lbs'
  ```

  - Formating the result using `next_prompt = "Observation: {}".format(result)` and passing it to the agent `abot(next_prompt)`, the result is now:
  ```
  'Answer: A Toy Poodle weighs an average of 7 lbs.'
  ```

  - All of the above steps are then added to a loop to get the result as requested.
  ```
  # python regular expression to selection action
  action_re = re.compile('^Action: (\w+): (.*)$')

  def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
      i += 1
      result = bot(next_prompt)
      print(result)
      actions = [
        action_re.match(a) 
        for a in result.split('\n') 
        if action_re.match(a)
      ]
      if actions:
        # There is an action to run
        action, action_input = actions[0].groups()
        if action not in known_actions:
            raise Exception("Unknown action: {}: {}".format(action, action_input))
        print(" -- running {} {}".format(action, action_input))
        observation = known_actions[action](action_input)
        print("Observation:", observation)
        next_prompt = "Observation: {}".format(observation)
      else:
        return
  ```

  - The entire result can now be acquired with just a question.
  ```
  question = """I have 2 dogs, a border collie and a scottish terrier. What is their combined weight"""
  query(question)
  ```
  The result obtained is:
  ```
  Thought: I need to find the average weight of both a Border Collie and a Scottish Terrier, then add them together to get the combined weight.
  Action: average_dog_weight: Border Collie
  PAUSE
  -- running average_dog_weight Border Collie
  Observation: a Border Collies average weight is 37 lbs
  Action: average_dog_weight: Scottish Terrier
  PAUSE
  -- running average_dog_weight Scottish Terrier
  Observation: Scottish Terriers average 20 lbs
  Thought: Now that I have the average weights of both dogs, I can calculate their combined weight by adding the two values together.
  Action: calculate: 37 + 20
  PAUSE
  -- running calculate 37 + 20
  Observation: 57
  Answer: The combined weight of a Border Collie and a Scottish Terrier is 57 lbs.
  ```
---

## LangGraph Components

- LangChain Prompt templates allow for the reuse of prompts by the agent.

- LangChain aloso has built-in tools that can be used, such as search.

- **LangGraph** helps in describing and orchestrating a control flow.
  - Allows for cyclic graphs.
  - Has Persistance.
    - Provides the ability to have multiple conversations at once.
    - Enables remembrance of previous conversations and actions.
    - Enables human-in-the-loop features.
  
- LangGraph is an extension on LangChain that supports graphs.
- Single and Multi-Agent flows are described and represented as graphs.
- Allows for extremely controlled flows.

- Graphs consist of:
  - *Nodes:* Agents or functions.
  - *Edges:* Connection between the nodes.
  - *Conditional Edges:* Decisions that can be made.
  - *Entrypoint:* Starting Node.
  - *End Node:* The action available to take after the agent.

- The state tracked over time is an important part of LangGraph.
  - AKA 'Agent State'.
  - It is accessible at all parts of the graph.
  - Local to the graph.
  - Can be stored in the persistance layer.
  - Two Types:
    - *Simple State:* a list of messages.
      ```
      class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
      ```
      When the state is updated with new messages, the existing message is not overwritten but is instead added to the state.
    - *Complex State:*
      ```
      class AgentState(TypedDict):
        input: str
        chat_history: list[BaseMessage]
        agent_outcome: Union[AgentAction, AgentFinish, None]
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
      ```
  
  - Example:
  ```
  # importing dependencies
  from langgraph.graph import StateGraph, END
  from typing import TypedDict, Annotated
  import operator
  from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
  from langchain_openai import ChatOpenAI
  from langchain_community.tools.tavily_search import TavilySearchResults

  # Search tool
  tool = TavilySearchResults(max_results=4) #increased number of results

  # State of the agent
  class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
  
  # Agent class
  class Agent:

    # Initializing the class
    def __init__(self, model, tools, system=""):
      self.system = system
      graph = StateGraph(AgentState)
      graph.add_node("llm", self.call_openai)
      graph.add_node("action", self.take_action)
      graph.add_conditional_edges(
          "llm",
          self.exists_action,
          {True: "action", False: END}
      )
      graph.add_edge("action", "llm")
      graph.set_entry_point("llm")
      self.graph = graph.compile()
      self.tools = {t.name: t for t in tools}
      self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
      result = state['messages'][-1]
      return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
      messages = state['messages']
      if self.system:
          messages = [SystemMessage(content=self.system)] + messages
      message = self.model.invoke(messages)
      return {'messages': [message]}

    def take_action(self, state: AgentState):
      tool_calls = state['messages'][-1].tool_calls
      results = []
      for t in tool_calls:
          print(f"Calling: {t}")
          if not t['name'] in self.tools:      # check for bad tool name from LLM
              print("\n ....bad tool name....")
              result = "bad tool name, retry"  # instruct LLM to retry if bad
          else:
              result = self.tools[t['name']].invoke(t['args'])
          results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
      print("Back to the model!")
      return {'messages': results}
  
  prompt = """You are a smart research assistant. Use the search engine to look up information. \
  You are allowed to make multiple calls (either together or in sequence). \
  Only look up information when you are sure of what you want. \
  If you need to look up some information before asking a follow up question, you are allowed to do that!
  """

  model = ChatOpenAI(model="gpt-3.5-turbo")
  abot = Agent(model, [tool], system=prompt)

  messages = [HumanMessage(content="What is the weather in sf?")]
  result = abot.graph.invoke({"messages": messages})

  result['messages'][-1].content
  ```
---

## Agentic Search Tool

- **Basic Search Tool Implementation:**
  - *Understand the query.*
    - Divide it into sub-questions if required - allows for understanding complex questions.
  - *Find the best source for each of the questions that the tool has to answer.*
  - *Extract the relevent information.*
    - Basic implementation is to obtain chunks and filter out the less relevent information.

- Example of Agentic Search:
  ```
  from dotenv import load_dotenv
  import os
  from tavily import TavilyClient

  # load environment variables from .env file
  _ = load_dotenv()

  # connect
  client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

  # run search
  result = client.search("What is in Nvidia's new Blackwell GPU?", include_answer=True)

  # print the answer
  result["answer"]
  ```

- Example of Regular Search:
  ```
  # Example query
  city = "San Francisco"

  query = f"""
    what is the current weather in {city}?
    Should I travel there today?
    "weather.com"
  """

  import requests
  from bs4 import BeautifulSoup
  from duckduckgo_search import DDGS
  import re

  ddg = DDGS()

  def search(query, max_results=6):
    try:
      results = ddg.text(query, max_results=max_results)
      return [i["href"] for i in results]
    except Exception as e:
      print(f"returning previous results due to exception reaching ddg.")
      results = [ # cover case where DDG rate limits due to high deeplearning.ai volume
        "https://weather.com/weather/today/l/USCA0987:1:US",
        "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
      ]
        return results  


  for i in search(query):
    print(i)

  def scrape_weather_info(url):
    """Scrape content from the given URL"""
    if not url:
      return "Weather information could not be found."
    
    # fetch data
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
      return "Failed to retrieve the webpage."

    # parse result
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup
  ```
  Printing the entire result:
  ```
  # use DuckDuckGo to find websites and take the first result
  url = search(query)[0]

  # scrape first wesbsite
  soup = scrape_weather_info(url)

  print(f"Website: {url}\n\n")
  print(str(soup.body)[:50000]) # limit long outputs
  ```
  Extracting the data:
  ```
  weather_data = []
  for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
      text = tag.get_text(" ", strip=True)
      weather_data.append(text)

  # combine all elements into a single string
  weather_data = "\n".join(weather_data)

  # remove all spaces from the combined text
  weather_data = re.sub(r'\s+', ' ', weather_data)
      
  print(f"Website: {url}\n\n")
  print(weather_data)
  ```

- Agentic Search Example:
  ```
  # run search
  result = client.search(query, max_results=1)

  # print first result
  data = result["results"][0]["content"]

  print(data)
  ```
  Print in a json format:
  ```
  import json
  from pygments import highlight, lexers, formatters

  # parse JSON
  parsed_json = json.loads(data.replace("'", '"'))

  # pretty print JSON with syntax highlighting
  formatted_json = json.dumps(parsed_json, indent=4)
  colorful_json = highlight(
    formatted_json,
    lexers.JsonLexer(),
    formatters.TerminalFormatter())

  print(colorful_json)
  ```
---

## Persistance and Streaming

- **Persistance** allows to store the state of the agent enabling to go back to a certain state at a later point in time.

- **Streaming** emits a list of signals of what is happening at that moment.

- Example Code:
  ```
  from dotenv import load_dotenv

  _ = load_dotenv()

  from langgraph.graph import StateGraph, END
  from typing import TypedDict, Annotated
  import operator
  from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
  from langchain_openai import ChatOpenAI
  from langchain_community.tools.tavily_search import TavilySearchResults

  tool = TavilySearchResults(max_results=2)

  class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

  from langgraph.checkpoint.sqlite import SqliteSaver

  memory = SqliteSaver.from_conn_string(":memory:")

  class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
      self.system = system
      graph = StateGraph(AgentState)
      graph.add_node("llm", self.call_openai)
      graph.add_node("action", self.take_action)
      graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
      graph.add_edge("action", "llm")
      graph.set_entry_point("llm")
      self.graph = graph.compile(checkpointer=checkpointer)
      self.tools = {t.name: t for t in tools}
      self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
      messages = state['messages']
      if self.system:
        messages = [SystemMessage(content=self.system)] + messages
      message = self.model.invoke(messages)
      return {'messages': [message]}

    def exists_action(self, state: AgentState):
      result = state['messages'][-1]
      return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
      tool_calls = state['messages'][-1].tool_calls
      results = []
      for t in tool_calls:
        print(f"Calling: {t}")
        result = self.tools[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
      print("Back to the model!")
      return {'messages': results} 
  ```
  In the agent class, to deal with persistance, a checkpointer is added into LangGraph which basically checkpoints the state after and between every node.
    - Using SqliteSaver, a simple check pointer that uses Sqlite - a built-in database under the hood.
      - **NOTE:** Any databse can be used for this, such as Redis and Postgres.
    - The initialized check pointer is passed into `graph.complie`.

  ```
  prompt = """You are a smart research assistant. Use the search engine to look up information. \
  You are allowed to make multiple calls (either together or in sequence). \
  Only look up information when you are sure of what you want. \
  If you need to look up some information before asking a follow up question, you are allowed to do that!
  """
  model = ChatOpenAI(model="gpt-4o")
  abot = Agent(model, [tool], system=prompt, checkpointer=memory)
  ```

  Adding Streaming - noting down the AI message that determines which action to take and the observation message that represents the result.
    - Adding the concept of thread config to keep track of different treads inside the persistant checkpointer.
      - Allows for multiple conversations at once.
    - Calling thr graph with `stream` instead of `invoke`.
      - passing in the messages and the thread.
      - Returns a stream of events that represent the updates to that state overtime.
  ```
  messages = [HumanMessage(content="What is the weather in sf?")]
  thread = {"configurable": {"thread_id": "1"}}

  for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
      print(v['messages'])

  messages = [HumanMessage(content="What about in la?")]
  thread = {"configurable": {"thread_id": "1"}}
  for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
      print(v)
  ```

- Streaming Tokens:
  ```
  from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

  memory = AsyncSqliteSaver.from_conn_string(":memory:")
  abot = Agent(model, [tool], system=prompt, checkpointer=memory)

  messages = [HumanMessage(content="What is the weather in SF?")]
  thread = {"configurable": {"thread_id": "4"}}
  async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
      content = event["data"]["chunk"].content
      if content:
        # Empty content in the context of OpenAI means
        # that the model is asking for a tool to be invoked.
        # So we only print non-empty content
        print(content, end="|")
  ```