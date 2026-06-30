from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Define the model
llm = HuggingFaceEndpoint(
    # repo_id="Qwen/Qwen2.5-7B-Instruct",
    # repo_id="meta-llama/Llama-3.1-70B-Instruct",
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    timeout=300,      # 5 minutes
    max_new_tokens=150,
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic':'cricket'})
print(result)

chain.get_graph().print_ascii()