import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pdfplumber
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Predefined path to the PDF file
pdf_path = 'Sudarshan Saur Shakti Pvt.pdf'

def extract_text_from_pdf(pdf_path):
    st.write("in extract_text_from_pdf")

    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    st.write("in get text chunks")
    """Split the extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=1
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save the vector store."""
    st.write("in get vector store")
    
    try:
        # Create embeddings object
        st.write(f"Number of text chunks: {len(text_chunks)}")
        st.write(f"First chunk preview: {text_chunks[0][:500]}")  # Preview of the first chunk
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.write("Embeddings created.")
        
        # Create FAISS vector store from the text chunks
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        st.write("Vector store created.")
        
        # Save the vector store to disk
        st.write("Saving vector store locally...")
        vector_store.save_local("faiss_index")
        st.success("Vector store saved successfully.")
        
    except Exception as e:
        st.error(f"Error creating or saving vector store: {e}")



def load_vector_store():
    st.write("in load_vector_store")
    """Load an existing vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

def get_conversational_chain():
    st.write("in get_conversational_chain")
    """Create the conversational chain."""
    prompt_template = """
    You are an intelligent assistant tasked with answering user questions as comprehensively as possible.

    Use the provided context from the PDF to generate a detailed and well-structured answer. If relevant details are missing from the context, provide a plausible explanation or additional related information to add value.

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    st.write("in user_input")
    """Process user input and provide responses."""
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(user_question, k=3)

    # Combine chunks into a single context
    combined_context = " ".join([doc.page_content for doc in docs])

    # Add documents to input
    input_documents = docs  # Pass the actual documents, not just the combined text

    chain = get_conversational_chain()
    response = chain({"input_documents": input_documents, "question": user_question}, return_only_outputs=True)

    if not response["output_text"].strip() or "answer is not available" in response["output_text"]:
        st.warning("The context does not contain sufficient details. Generating an additional answer...")
        secondary_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
        secondary_response = secondary_model({"question": user_question})
        st.write("Additional Insights: ", secondary_response["output_text"])

    st.write("Reply: ", response["output_text"])


def main():
    """Main Streamlit app."""
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Automatically extract text from the predefined PDF
    raw_text = extract_text_from_pdf(pdf_path)
    st.write("Extracted Text:", raw_text[:1000])  # Displaying a small portion of the extracted text for debugging
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    st.success("PDF processed and vector store created.")

    # User input to query the PDF
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
