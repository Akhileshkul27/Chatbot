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

# CSS styling for the page
st.markdown("""
    <style>
        body {
            background-color: #f4f6f9;
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            color: #333;
            padding: 10px;
            font-size: 28px;
            font-weight: bold;
        }
        .chat-box {
            background-color: #fff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .ai-response {
            background-color: #e0f7fa;
            padding: 10px;
            border-radius: 5px;
            font-size: 16px;
            color: #00796b;
            margin-bottom: 15px;
        }
        .user-query {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 5px;
            font-size: 16px;
            color: #004d40;
            margin-bottom: 15px;
        }
        .submit-btn {
            background-color: #00796b;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
        }
        .submit-btn:hover {
            background-color: #004d40;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #777;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the extracted text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save the vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_vector_store():
    """Load an existing vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

def get_conversational_chain():
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

    st.markdown(f"<div class='ai-response'>{response['output_text']}</div>", unsafe_allow_html=True)

def main():
    """Main Streamlit app."""
    st.markdown("<div class='header'>AI Assistant for Business Insights</div>", unsafe_allow_html=True)

    # Automatically extract text from the predefined PDF
    raw_text = extract_text_from_pdf(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    st.success("PDF processed and vector store created.")

    # User input to query the PDF
    user_question = st.text_input("Ask a Business Question", key="question")
    if user_question:
        st.markdown(f"<div class='user-query'>{user_question}</div>", unsafe_allow_html=True)
        user_input(user_question)

    # Footer
    st.markdown("<div class='footer'>Powered by AI - Sudarshan Saur Shakti Pvt. Ltd.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
