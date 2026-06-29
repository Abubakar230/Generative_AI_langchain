from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Generate an embedding for this text using OpenAI's embedding model.",
    "Generative AI is transforming the way we interact with technology.",
    "Embeddings are a powerful tool for natural language processing tasks."
]

vector = embedding.embed_documents(documents)

print(str(vector))

