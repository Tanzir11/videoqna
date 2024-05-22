import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import json
import os

st.title("Question Answering System")

class QnaRequest:
    def __init__(self, questions, chat_history):
        self.questions = questions
        self.chat_history = chat_history

origins = ["*"]

prompts = """You're given the transcripts of 3 videos your job is to answer the questions based on the provided context which is the transcripts.
Context: {context}
question: {question}
Remember you have the chat_history,
answer what has been asked.
for the first query by no other query will be available so greet for the first on.
Answer question based only on question parameters which are given.
Make sure to answer what has been asked.
"""

os.environ["OPENAI_API_KEY"] = ""

embeddings = OpenAIEmbeddings()

def Qna(questions, user_queries):
    db_instructEmbedd = FAISS.load_local("faiss_index", embeddings)
    retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 3})
    temp1 = prompts + " " + "The previous queries that the user has made are as follows, use this to answer questions" + " " + user_queries
    prompt = ChatPromptTemplate.from_template(temp1)
    model = ChatOpenAI(model="gpt-4", temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
        )

    data = chain.invoke(questions)
    return data

def extract_data(input_data):
    user_queries = []
    for i in range(len(input_data)):
        user_queries.append(input_data[i])
    return user_queries

def main():
    st.subheader("Ask your question:")
    questions = st.text_area("Enter your question here:")
    chat_history = st.text_area("Enter chat history (separated by new lines):")
    chat_history_list = chat_history.split("\n")
    user_queries = extract_data(chat_history_list)

    if st.button("Submit"):
        try:
            request = QnaRequest(questions, user_queries)
            result = Qna(request.questions, json.dumps(request.chat_history))
            st.write("Response:")
            # Displaying the result within brackets
            st.write(f"[{result}]")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
