import streamlit as st
from dotenv import load_dotenv
from pyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.basa import ConversationalRetrievalChain
from langchain.memory import ConversationalBufferMemory

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def get_pdf_text(filename= "bio.pdf"):
    text = ""
    try:
        pdf_reader = PdfReader(filename)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as p:
        print(f"There is an error in {filename}: {p}")


def text_to_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts = text_chunks, embeddings = embeddings)
    return vector_store


def get_conversation(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
            return True
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm = llm,
        memory = memory
        retriever = vector_store.as_retriever(),
    )
    return conversation


def handles_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']


    for i, message in enumerate(st.session_state.chat_history):
        if i %2 == 0:
            st.write(f"User question : {message.content}")
        else:
            st.write(f"Bot answer : {message.content}")


def main():
    load_dotenv()

    st.set_page_config(page_title="Portfolio", page_icon= ":books")
    st.write("Ask questions about me from BUddy")

    if 'conversation' not in st.session_state:
        raw_text = get_pdf_text()
        text_chunks = text_to_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        st.session_state.conversation = get_conversation(vector_store)


    user_question = st.text_input("Type your question here:")
    if user_question:
        handles_user_input(user_question)


if __name__ == '__main__':
    main()



    


