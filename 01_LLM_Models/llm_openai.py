# https://platform.openai.com/  get openai api key from this with min 5$
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# llm = OpenAI(model="gpt-4")
llm = OpenAI(model="gpt-3.5-turbo-instruct")  # Use GPT-3.5-turbo model

result = llm.invoke("Write a short poem about the beauty of nature.")
print(result)