import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
import base64
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import json

load_dotenv()
api_key = 'YOUR_API_KEY'  # Replace with your actual API key
genai.configure(api_key=api_key)

# Function to set background image
def set_background_image(image_path):
    try:
        image_base64 = get_image_base64(image_path)
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{image_base64}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.error(f"Error: Background image not found at '{image_path}'")
    except Exception as e:
        st.error(f"An error occurred while setting the background image: {e}")

# Function to encode image as base64 for inline usage
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at '{image_path}'")
    except Exception as e:
        raise Exception(f"Error encoding image: {e}")

def get_pdf_text(pdf_docs):
    text = ""
    try:
        if isinstance(pdf_docs, list):
            for pdf in pdf_docs:
                pdf_file = io.BytesIO(pdf.read())
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        else:
            pdf_file = io.BytesIO(pdf_docs.read())
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error processing PDF files: {e}")
        return None
    return text

def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        return

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating or saving vector store: {e}")

def get_conversational_chain():
    try:
        prompt_template = """You are a AI Chatbot , please start with Greetings in your First response ,Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say "Please ask Questions Related to Zurich:", don't provide wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error initializing conversational chain: {e}")
        return None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def user_input(question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(question)
        chain = get_conversational_chain()

        if chain:
            response = chain({"input_documents": docs, "question": question})
            output_text = response['output_text']
            return output_text
        else:
            st.error("Conversational chain could not be initialized.")
            return None
    except Exception as e:
        st.error(f"Error processing user input: {e}")
        return None

# Load Lottie animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_loading = load_lottiefile("assets/Animation - 1742736638113.json")  # Replace with your Lottie file

def main():
    st.set_page_config(
        page_title="Zurich_Agent",
        page_icon="	:black_circle:",
        initial_sidebar_state="collapsed"
    )
    image_path = "assets/bg2.jpg"
    set_background_image(image_path)
    st.header("Zurich GenAI Agent")

    # Store a key for the animation container
    animation_container = st.empty()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display Lottie animation
        with animation_container.container():
            col1, col2 = st.columns([5, 1])
            with col2:
                st_lottie(
                    lottie_loading,
                    key="loading",
                    loop=True,
                    width=100,
                    height=100
                )

        # Get the response
        response = user_input(prompt)

        # Clear the animation
        animation_container.empty()

        # Display the response
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        get_vector_store(text_chunks)
                        st.success("Done")
                    else:
                        st.error("PDF text chunking failed.")
                else:
                    st.error("PDF text extraction failed.")
            else:
                st.warning("Please upload PDF files.")

if __name__ == "__main__":
    main()