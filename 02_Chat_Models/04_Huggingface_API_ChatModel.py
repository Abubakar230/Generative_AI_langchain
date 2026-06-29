from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # repo_id="Qwen/Qwen2.5-7B-Instruct",
    repo_id="LiquidAI/LFM2.5-230M",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a short poem about the beauty of nature.")
print(result.content)
