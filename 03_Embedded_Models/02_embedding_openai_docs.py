from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Generate an embedding for this text using OpenAI's embedding model.",
    "Generative AI is transforming the way we interact with technology.",
    "Embeddings are a powerful tool for natural language processing tasks."
]

result = embedding.embed_documents(documents)

print(str(result))