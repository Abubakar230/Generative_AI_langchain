from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=50)  # Use GPT-4 model with a temperature of 0.7

result = model.invoke("Write a short poem about the beauty of nature.")
print(result.content)