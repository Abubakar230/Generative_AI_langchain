from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    # repo_id="meta-llama/Llama-3.3-70B-Instruct",
    # repo_id="meta-llama/Llama-3.1-70B-Instruct",
    task="text-generation",
    # timeout=300,      # 5 minutes
    # max_new_tokens=100,
)
model1 = ChatHuggingFace(llm=llm1)
llm2 = HuggingFaceEndpoint(
    # repo_id="Qwen/Qwen2.5-7B-Instruct",
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    # repo_id="meta-llama/Llama-3.1-70B-Instruct",
    task="text-generation",
    # timeout=300,      # 5 minutes
    # max_new_tokens=100,
)
model2 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model1, parser),
    'linkedin': RunnableSequence(prompt2, model2, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedin'])