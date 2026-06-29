from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)



st.header("HuggingFace Chat Model")

user_input = st.text_input("Enter your prompt:")
if st.button('Sumarize'):
    result = model.invoke(user_input)
    st.write(result.content)



# result = model.invoke("Write a short poem about the beauty of nature.")
# print(result.content)