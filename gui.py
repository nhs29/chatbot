import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate


st.set_page_config(page_title="Document Genie", layout="wide")
st.markdown("""## Document Genie: Get instant insights from your Documents...""")

vectorstore = FAISS.load_local("faiss_store (4)", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

template = """You are a helpful assistant.
Use the following pieces of context, if needed, to answer the input at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
\n\nContext: {context}
\n\nChat history: {chat_history}
New question: {question}
Helpful answer:"""
prompt = PromptTemplate.from_template(template)

llm = OpenAI(temperature=0.5)

memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", output_key='answer', return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": prompt}, verbose=True, rephrase_question=False)


def ask(question):
    response = chain({"question": question})
    st.write(response["answer"].strip())


if __name__ == "__main__":
    key = st.text_input("API key")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        question = st.text_input("Ask a question")
        if question:
            ask(question)
