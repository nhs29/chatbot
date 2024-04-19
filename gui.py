import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
llm = OpenAI(temperature=0.5)

vectorstore = FAISS.load_local("faiss_store (4)", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

template = """You are a helpful assistant.
    Use the following pieces of context, if needed, to answer the input at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    \n\nContext: {context}
    \n\nChat history: {chat_history}
    New question: {question}
    Helpful answer:"""
prompt = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", output_key='answer', return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory,
                                              combine_docs_chain_kwargs={"prompt": prompt}, verbose=True,
                                              rephrase_question=False)

st.set_page_config(page_title="History Chatbot")
st.markdown("""## History Chatbot""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
else:
    for message in st.session_state.chat_history:
        memory.save_context({"question": message["human"]}, {"answer": message["assistant"]})


question = st.chat_input("Ask a question")
if question:
    response = chain({"question": question})
    message = {"human": question, "assistant": response["answer"]}
    st.session_state.chat_history.append(message)
    for messages in st.session_state.chat_history:
        with st.chat_message("human"):
            st.write(messages["human"])
        with st.chat_message("assistant"):
            st.write(messages["assistant"])

