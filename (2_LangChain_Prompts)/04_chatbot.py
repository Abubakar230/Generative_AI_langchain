from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

# Save the chat history in a list of messages that who is AI and who is human. The first message is a system message that sets the context for the conversation. The second message is a human message that asks the AI to tell them about LangChain. The third message is an AI message that contains the AI's response to the human's question.
# So we can easily differentiate between the messages and also we can easily add new messages to the chat history. The chat history is then passed to the model.invoke() method, which generates a response based on the entire conversation history.
chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)