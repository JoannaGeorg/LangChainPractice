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