# (Both prompts prints)first Jokes prints then prints the explanation
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    # repo_id="meta-llama/Llama-3.3-70B-Instruct",
    # repo_id="meta-llama/Llama-3.1-70B-Instruct",
    task="text-generation",
    # timeout=300,      # 5 minutes
    # max_new_tokens=100,
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

print(result)