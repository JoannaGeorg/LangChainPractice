# LangChain For LLM Application Development

## Overview

- Open-source developmet framework for LLMs.
- Python and Javascript Packages.
- Focused on composition and modularity.

## Models, Prompts, and Parsers

- Models refer to the language model underpining the project.
  
  ```
  #!pip install --upgrade langchain
  ```

  ```
  from langchain.chat_models import ChatOpenAI
  chat = ChatOpenAI(temperature=0.0, model=llm_model)
  chat
  ```

- Prompts is the style of creating inputs to pass into the models.

  ```
  template_string = """Translate the text that is delimited by triple backticks \
  into a style that is {style}. \
  
  text: \`\`\`{text}\`\`\`
  """
  ```

  ```
  from langchain.prompts import ChatPromptTemplate
  prompt_template = ChatPromptTemplate.from_template(template_string)
  ```

  Checking the prompt template:
  ```
  prompt_template.messages[0].prompt
  ```

  Obtaining the input variables for the prompt:
  ```
  prompt_template.messages[0].prompt.input_variables
  ```

  Prompting the LLM:
  ```
  customer_style = """American English \
  in a calm and respectful tone
  """

  customer_email = """
  Arrr, I be fuming that me blender lid \
  flew off and splattered me kitchen walls \
  with smoothie! And to make matters worse, \
  the warranty don't cover the cost of \
  cleaning up me kitchen. I need yer help \
  right now, matey!
  """

  customer_messages = prompt_template.format_messages(
    style=customer_style,
    text=customer_email
  )

  print(customer_messages[0])

  # Call the LLM to translate to the style of the customer message
  customer_response = chat(customer_messages)

  print(customer_response.content)
  ```

- Parsers involves taking the output of the models and parsing it through a more structured format.

  Define the format of the output:
  ```
  {
    "gift": False,
    "delivery_days": 5,
    "price_value": "pretty affordable!"
  }
  ```
  To get this format:
  ```
  customer_review = """\
  This leaf blower is pretty amazing.  It has four settings:\
  candle blower, gentle breeze, windy city, and tornado. \
  It arrived in two days, just in time for my wife's \
  anniversary present. \
  I think my wife liked it so much she was speechless. \
  So far I've been the only one using it, and I've been \
  using it every other morning to clear the leaves on our lawn. \
  It's slightly more expensive than the other leaf blowers \
  out there, but I think it's worth it for the extra features.
  """

  review_template = """\
  For the following text, extract the following information:

  gift: Was the item purchased as a gift for someone else? \
  Answer True if yes, False if not or unknown.

  delivery_days: How many days did it take for the product \
  to arrive? If this information is not found, output -1.

  price_value: Extract any sentences about the value or price,\
  and output them as a comma separated Python list.

  Format the output as JSON with the following keys:
  gift
  delivery_days
  price_value

  text: {text}
  """
  ```

  Passing this through the LLM:
  ```
  from langchain.prompts import ChatPromptTemplate

  prompt_template = ChatPromptTemplate.from_template(review_template)
  print(prompt_template)

  messages = prompt_template.format_messages(text=customer_review)
  chat = ChatOpenAI(temperature=0.0, model=llm_model)
  response = chat(messages)
  print(response.content)
  ```

  The respomse type of `response.content` is that of a string. Thus, it need to be parsed into a python dictionary.

  ```
  from langchain.output_parsers import ResponseSchema
  from langchain.output_parsers import StructuredOutputParser

  gift_schema = ResponseSchema(
    name="gift",
    description="Was the item purchased as a gift for someone else? \
    Answer True if yes, False if not or unknown."
  )
  delivery_days_schema = ResponseSchema(
    name="delivery_days",
    description="How many days did it take for the product\
    to arrive? If this information is not found, output -1."
  )
  price_value_schema = ResponseSchema(
    name="price_value",
    description="Extract any sentences about the value or price, and output them as a \
    comma separated Python list."
  )

  response_schemas = [
    gift_schema, 
    delivery_days_schema,
    price_value_schema
  ]

  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  format_instructions = output_parser.get_format_instructions()

  review_template_2 = """\
  For the following text, extract the following information:

  gift: Was the item purchased as a gift for someone else? \
  Answer True if yes, False if not or unknown.

  delivery_days: How many days did it take for the product\
  to arrive? If this information is not found, output -1.

  price_value: Extract any sentences about the value or price,\
  and output them as a comma separated Python list.

  text: {text}

  {format_instructions}
  """

  prompt = ChatPromptTemplate.from_template(template=review_template_2)

  messages = prompt.format_messages(
    text=customer_review, 
    format_instructions=format_instructions
  )

  response = chat(messages)
  print(response.content)

  output_dict = output_parser.parse(response.content)
  print(type(output_dict))
  ```

## Memory

An LLM does not organically have a memeory. There are specific methods through which one can simulate having a memeory. The following are a few of the available ways to incorporate conversation history to an LLM.

#### ConversationBufferMemory

To save the history of a conversion.

  ```
  from langchain.chat_models import ChatOpenAI
  from langchain.chains import ConversationChain
  from langchain.memory import ConversationBufferMemory

  llm = ChatOpenAI(temperature=0.0, model=llm_model)
  memory = ConversationBufferMemory()
  conversation = ConversationChain(
      llm=llm, 
      memory = memory,
      verbose=True
  )
  ```

Examples of a conversation.
  ```
  conversation.predict(input="Hi, my name is Andrew")
  conversation.predict(input="What is 1+1?")
  conversation.predict(input="What is my name?")
  ```
The conversation history allows the LLM to answer the 3rd question correctly.

To see the memory buffer `print(memory.buffer)`.

To see the load emmory variables `memory.load_memory_variables({})`.

Creating one's own conversation history:
  ```
  memory = ConversationBufferMemory()
  memory.save_context({"input": "Hi"}, {"output": "What's up"})
  memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
  print(memory.load_memory_variables({}))
  ```

#### ConversationBufferWindowMemory

Limit the amount of conversation history the LLM has to a specific window.

For Example:
  ```
  from langchain.memory import ConversationBufferWindowMemory

  memory = ConversationBufferWindowMemory(k=1)
  ```
  This limits the memory to one previous conversation. Thus, when the foloowing code is run:
  ```
  memory.save_context({"input": "Hi"}, {"output": "What's up"})
  memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
  ```
  The resulting print of `memory.load_memory_variables({})` is only the second conversation. Therefore, given the chat example in the previous lesson, Asking chat to tell the name no longer works as it is a conversation that was before the previous one.

  Changing the value of k will result in different number of chat history that is remembered.

#### ConversationTokenBufferMemory

Memory is in tokens and can be limited.

Example code:
  ```
  #!pip install tiktoken

  from langchain.memory import ConversationTokenBufferMemory
  from langchain.llms import OpenAI
  llm = ChatOpenAI(temperature=0.0, model=llm_model)

  memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
  memory.save_context({"input": "AI is what?!"}, {"output": "Amazing!"})
  memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful!"})
  memory.save_context({"input": "Chatbots are what?"}, {"output": "Charming!"})

  print(memory.load_memory_variables({}))
  ```

#### ConversationSummaryMemory

When the history crosses the limit placed, the LLM can create a summary of the conversation history to pass as context in the prompt.

Example:
  ```
  from langchain.memory import ConversationSummaryBufferMemory

  # create a long string
  schedule = "There is a meeting at 8am with your product team. \
  You will need your powerpoint presentation prepared. \
  9am-12pm have time to work on your LangChain \
  project which will go quickly because Langchain is such a powerful tool. \
  At Noon, lunch at the italian resturant with a customer who is driving \
  from over an hour away to meet you to understand the latest in AI. \
  Be sure to bring your laptop to show the latest LLM demo."

  memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
  memory.save_context({"input": "Hello"}, {"output": "What's up"})
  memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
  memory.save_context({"input": "What is on the schedule today?"}, {"output": f"{schedule}"})
  ```

  The history is too long and so, the stored history is a summary when `memory.load_memory_variables({})` is printed.

  princting:
  ```
  conversation = ConversationChain(
      llm=llm, 
      memory = memory,
      verbose=True
  )

  conversation.predict(input="What would be a good demo to show?")
  ```

## Chains

The chain combines an LLM together with a prompt which is used as a bulding block that can be combined with other building blocks to carry out a sequence of operations.

#### LLMChain

Code:
  ```
  from langchain.chat_models import ChatOpenAI
  from langchain.prompts import ChatPromptTemplate
  from langchain.chains import LLMChain

  llm = ChatOpenAI(temperature=0.9, model=llm_model)

  prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
  )

  chain = LLMChain(llm=llm, prompt=prompt)

  product = "Queen Size Sheet Set"
  chain.run(product)
  ```

#### SimpleSequenceChain

Code Example:
  ```
  from langchain.chains import SimpleSequentialChain

  llm = ChatOpenAI(temperature=0.9, model=llm_model)

  # prompt template 1
  first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
  )

  # Chain 1
  chain_one = LLMChain(llm=llm, prompt=first_prompt)

  # prompt template 2
  second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company:{company_name}"
  )
  # chain 2
  chain_two = LLMChain(llm=llm, prompt=second_prompt)

  overall_simple_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
  )

  overall_simple_chain.run(product)
  ```

#### SequentialChain

Code Example:
  ```
  from langchain.chains import SequentialChain

  llm = ChatOpenAI(temperature=0.9, model=llm_model)

  # prompt template 1: translate to english
  first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english: \n\n{Review}"
  )
  # chain 1: input= Review and output= English_Review
  chain_one = LLMChain(
    llm=llm,
    prompt=first_prompt, 
    output_key="English_Review"
  )

  second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence: \n\n{English_Review}"
  )
  # chain 2: input= English_Review and output= summary
  chain_two = LLMChain(
    llm=llm, 
    prompt=second_prompt, 
    output_key="summary"
  )

  # prompt template 3: translate to english
  third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
  )
  # chain 3: input= Review and output= language
  chain_three = LLMChain(
    llm=llm, 
    prompt=third_prompt,
    output_key="language"
  )

  # prompt template 4: follow up message
  fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
  )
  # chain 4: input= summary, language and output= followup_message
  chain_four = LLMChain(
    llm=llm,
    prompt=fourth_prompt,
    output_key="followup_message"
  )

  # overall_chain: input= Review 
  # and output= English_Review,summary, followup_message
  overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
  )

  review = df.Review[5]
  overall_chain(review)
  ```

#### RouterChain

This Chain has multiple destination chains, each doing a specific role. The type of prompt selects which role best filt for a response.

Code Example:
  Example Role Prompts:
  ```
  physics_template = """You are a very smart physics professor. \
  You are great at answering questions about physics in a concise\
  and easy to understand manner. \
  When you don't know the answer to a question you admit\
  that you don't know.

  Here is a question:
  {input}"""


  math_template = """You are a very good mathematician. \
  You are great at answering math questions. \
  You are so good because you are able to break down \
  hard problems into their component parts, 
  answer the component parts, and then put them together\
  to answer the broader question.

  Here is a question:
  {input}"""

  history_template = """You are a very good historian. \
  You have an excellent knowledge of and understanding of people,\
  events and contexts from a range of historical periods. \
  You have the ability to think, reflect, debate, discuss and \
  evaluate the past. You have a respect for historical evidence\
  and the ability to make use of it to support your explanations \
  and judgements.

  Here is a question:
  {input}"""
  ```

  Prompt Information:
  ```
  prompt_infos = [
    {
      "name": "physics", 
      "description": "Good for answering questions about physics", 
      "prompt_template": physics_template
    },
    {
      "name": "math", 
      "description": "Good for answering math questions", 
      "prompt_template": math_template
    },
    {
      "name": "History", 
      "description": "Good for answering history questions", 
      "prompt_template": history_template
    }
  ]
  ```

  Forming the chain:
  ```
  from langchain.chains.router import MultiPromptChain
  from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
  from langchain.prompts import PromptTemplate

  llm = ChatOpenAI(temperature=0, model=llm_model)

  destination_chains = {}
  for p_info in prompt_infos:
      name = p_info["name"]
      prompt_template = p_info["prompt_template"]
      prompt = ChatPromptTemplate.from_template(template=prompt_template)
      chain = LLMChain(llm=llm, prompt=prompt)
      destination_chains[name] = chain  
      
  destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
  destinations_str = "\n".join(destinations)

  default_prompt = ChatPromptTemplate.from_template("{input}")
  default_chain = LLMChain(llm=llm, prompt=default_prompt)

  MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
  language model select the model prompt best suited for the input. \
  You will be given the names of the available prompts and a \
  description of what the prompt is best suited for. \
  You may also revise the original input if you think that revising\
  it will ultimately lead to a better response from the language model.

  << FORMATTING >>
  Return a markdown code snippet with a JSON object formatted to look like:
  \`\`\`json
  {{{{
      "destination": string \ "DEFAULT" or name of the prompt to use in {destinations}
      "next_inputs": string \ a potentially modified version of the original input
  }}}}
  \`\`\`

  REMEMBER: The value of “destination” MUST match one of \
  the candidate prompts listed below.\
  If “destination” does not fit any of the specified prompts, set it to “DEFAULT.”
  REMEMBER: "next_inputs" can just be the original input \
  if you don't think any modifications are needed.

  << CANDIDATE PROMPTS >>
  {destinations}

  << INPUT >>
  {{input}}

  << OUTPUT (remember to include the \`\`\`json)>>"""

  router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
  )
  router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
  )

  router_chain = LLMRouterChain.from_llm(llm, router_prompt)

  chain = MultiPromptChain(
    router_chain=router_chain, 
    destination_chains=destination_chains, 
    default_chain=default_chain, verbose=True
  )

  # Physics based
  chain.run("What is black body radiation?")

  # Math Based
  chain.run("what is 2 + 2")
  ```

## Question and Answer

Given a document, or a piece of text, the LLM is expected to answer question on them.

Example Code:
  ```
  from langchain.chains import RetrievalQA
  from langchain.chat_models import ChatOpenAI
  from langchain.document_loaders import CSVLoader
  from langchain.vectorstores import DocArrayInMemorySearch
  from IPython.display import display, Markdown
  from langchain.llms import OpenAI

  file = 'OutdoorClothingCatalog_1000.csv'
  loader = CSVLoader(file_path=file)

  from langchain.indexes import VectorstoreIndexCreator

  index = VectorstoreIndexCreator(vectorstore_cls=DocArrayInMemorySearch).from_loaders([loader])

  query ="Please list all your shirts with sun protection in a table in markdown and summarize each one."

  llm_replacement_model = OpenAI(
    temperature=0, 
    model='gpt-3.5-turbo-instruct'
  )

  response = index.query(query, llm = llm_replacement_model)

  display(Markdown(response))
  ```

However, when faced with large documents, embeddings and vectors are used to combat the issues that arise.

Example Code:
  ```
  from langchain.document_loaders import CSVLoader
  loader = CSVLoader(file_path=file)

  docs = loader.load()

  from langchain.embeddings import OpenAIEmbeddings
  embeddings = OpenAIEmbeddings()

  embed = embeddings.embed_query("Hi my name is Harrison")

  db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
  )

  query = "Please suggest a shirt with sunblocking"
  docs = db.similarity_search(query)
  len(docs)

  retriever = db.as_retriever()
  llm = ChatOpenAI(temperature = 0.0, model=llm_model)  
  qdocs = "".join([docs[i].page_content for i in range(len(docs))])
  response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
  shirts with sun protection in a table in markdown and summarize each one.") 

  display(Markdown(response))
  ```