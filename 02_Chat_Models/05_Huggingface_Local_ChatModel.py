# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# import os

# os.environ['HF_HOME'] = 'D:/huggingface_cache'

# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text-generation',
#     pipeline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=100
#     )
# )
# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India")

# print(result.content)

from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="LiquidAI/LFM2.5-230M",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.7,
    },
)
print(llm.invoke("Define intro of GenAI."))
print(llm.invoke("Write a short poem about nature."))