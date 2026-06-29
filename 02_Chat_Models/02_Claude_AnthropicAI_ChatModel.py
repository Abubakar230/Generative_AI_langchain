# console.anthropic.com
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7, max_completion_tokens=50)  # Use Claude model with a temperature of 0.7

result = model.invoke("Write a short poem about the beauty of nature.")
print(result.content)