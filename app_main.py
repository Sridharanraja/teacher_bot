# import os
# import streamlit as st
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from groq import Groq

# # Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Initialize Groq client
# client = Groq(api_key=GROQ_API_KEY)

# MODEL_ID = "llama3-8b-8192"

# # Function to extract text from PDF files
# # def get_pdf_text(pdf_docs):
# #     text = ""
# #     for pdf in pdf_docs:
# #         reader = PdfReader(pdf)
# #         for page in reader.pages:
# #             content = page.extract_text()
# #             if content:
# #                 text += content
# #     return text

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:  # Now pdf is a file path
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# # Split text into chunks
# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return splitter.split_text(text)

# # Store vector index using FAISS
# def get_vector_store(chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Load vector store and retrieve relevant documents
# def get_relevant_docs(question):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
#     return db.similarity_search(question, k=4)

# # Use Groq API to generate response

# #If the answer is not available in the context, say: "The answer is not available in the context."

# def generate_response(context, question):
#     system_message = {
#         "role": "system",
#         "content": """
#         You are a helpful AI assistant that answers user questions using the provided context.
#         some times some of the questions answer is not availabe, So please tell the answer and explaination
#         Do not guess or fabricate.
#         """
#     }

#     user_message = {
#         "role": "user",
#         "content": f"""
#         Context:
#         {context}

#         Question: {question}
#         """
#     }

#     response = client.chat.completions.create(
#         model=MODEL_ID,
#         messages=[system_message, user_message],
#         temperature=0.3,
#         max_tokens=2048
#     )
#     return response.choices[0].message.content

# # Main Streamlit app
# def main():
#     st.set_page_config(page_title="Groq PDF Chatbot")
#     st.header("Groq-powered LLM Chatbot for PDF Files 🧠📄")

#     user_question = st.text_input("Ask a question about the uploaded PDFs:")

#     if user_question:
#         docs = get_relevant_docs(user_question)
#         context = "\n\n".join([doc.page_content for doc in docs])
#         with st.spinner("Thinking..."):
#             reply = generate_response(context, user_question)
#         st.write("### Reply:")
#         st.write(reply)



#     PDF_FOLDER = "E:/NEXOFY/Projects/Whatsapp_Bot/RAG-local/DATA/PDF"  # Change to your actual folder path

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

#         selected_files = st.multiselect("Select PDF files to process", pdf_files)

#         if st.button("Submit & Process"):
#             if selected_files:
#                 with st.spinner("Processing..."):
#                     file_paths = [os.path.join(PDF_FOLDER, f) for f in selected_files]
#                     raw_text = get_pdf_text(file_paths)
#                     text_chunks = get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Done")
#             else:
#                 st.warning("Please select at least one PDF file.")


#     # with st.sidebar:
#     #     st.title("Upload PDFs")
#     #     pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
#     #     if st.button("Submit & Process"):
#     #         with st.spinner("Reading and indexing..."):
#     #             text = get_pdf_text(pdf_docs)
#     #             chunks = get_text_chunks(text)
#     #             get_vector_store(chunks)
#     #             st.success("PDFs processed successfully!")

# if __name__ == "__main__":
#     main()



import os
import json
import streamlit as st
st.set_page_config(page_title="Teacher Assit Chatbot")

# st.set_page_config(page_title="PDF QA Login", page_icon="🔐", layout="centered")
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
import time

def load_users(json_path="credential.json"):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    else:
        return {}
    
# --- Login page ---
def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "just_logged_in" not in st.session_state:
        st.session_state.just_logged_in = False

    users = load_users()

    if not st.session_state.logged_in:
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.just_logged_in = True  # flag for temporary message
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
    else:
        # ✅ Show "logged in" success message only once for 2 seconds
        if st.session_state.just_logged_in:
            st.success(f"✅ Logged in as: {st.session_state.username}")
            time.sleep(1)
            st.session_state.just_logged_in = False
            st.rerun()
        else:
            st.sidebar.markdown(f"👤 Logged in as: `{st.session_state.username}`")
            if st.sidebar.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.rerun()

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)
MODEL_ID = "llama3-8b-8192"

PDF_FOLDER = "E:/NEXOFY/Projects/Whatsapp_Bot/RAG-local/DATA/PDF"  # Change to your actual folder path

# PDF_FOLDER = "PDF"  # folder containing your PDF files
INDEX_FOLDER = "faiss_indexes"  # where you store individual FAISS indexes

# Ensure index folder exists
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Extract text from PDF file
def get_pdf_text(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Split into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# Create and save FAISS index for a single PDF
def create_vector_index(pdf_path):
    text = get_pdf_text(pdf_path)
    chunks = get_text_chunks(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    save_path = os.path.join(INDEX_FOLDER, f"{base_name}_index")
    vector_store.save_local(save_path)

# Load vector store and retrieve relevant documents
def get_relevant_docs(index_path, question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return db.similarity_search(question, k=4)

# Generate answer using Groq
def generate_response(context, question):
    system_message = {
        "role": "system",
        "content": """
        You are a helpful AI teaching assistant. Answer user questions only using the information provided in the given context.
        If the answer cannot be found in the context, respond with: "The answer is not available in the context."
        Do not guess or fabricate.
        Interpret user queries flexibly, recognizing and matching words with similar meanings even if they differ slightly from the wording in the context.
        """
    }

    user_message = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    }

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[system_message, user_message],
        temperature=0.3,
        max_tokens=2048
    )
    return response.choices[0].message.content

# Main Streamlit App
def main_app():
    # st.set_page_config(page_title="Teacher Assit Chatbot")
    st.write(f"Welcome to the application, **{st.session_state.username}**!")
    st.header("Teacher Assit Chatbot 🧠📄")

    # # Sidebar - Select PDFs
    # with st.sidebar:
    #     st.title("Select PDFs")
    #     pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    #     selected_files = st.multiselect("Choose PDF files to ask questions from:", pdf_files)

    # Sidebar - Select a single PDF
    with st.sidebar:
        st.title("Select Subject")
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
        # selected_file = st.radio("Choose the Subject to ask questions from:", pdf_files)
        
        # Create a mapping from filename without extension to full filename
        file_name_map = {os.path.splitext(f)[0]: f for f in pdf_files}

        # Display only the names without .pdf in the radio options
        selected_name = st.radio("Choose the Subject to ask questions from:", list(file_name_map.keys()))

        # Get the full filename with .pdf extension
        selected_file = file_name_map[selected_name]

    user_question = st.text_input("Ask your doubts and question:")
    
    if user_question and selected_file:
        pdf_path = os.path.join(PDF_FOLDER, selected_file)
        base_name = os.path.splitext(selected_file)[0]
        index_path = os.path.join(INDEX_FOLDER, f"{base_name}_index")

        if not os.path.exists(index_path):
            with st.spinner(f"Indexing {selected_file}..."):
                create_vector_index(pdf_path)

        with st.spinner(f"Getting answer from {selected_file}..."):
            try:
                docs = get_relevant_docs(index_path, user_question)
                context = "\n\n".join([doc.page_content for doc in docs])
                answer = generate_response(context, user_question)
                st.write(f"### 📄 Answer from: {selected_file}")
                st.write(answer)
            except Exception as e:
                st.error(f"Error with {selected_file}: {str(e)}")

    # if user_question and selected_files:
    #     for pdf_file in selected_files:
    #         pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    #         base_name = os.path.splitext(pdf_file)[0]
    #         index_path = os.path.join(INDEX_FOLDER, f"{base_name}_index")

    #         if not os.path.exists(index_path):
    #             with st.spinner(f"Indexing {pdf_file}..."):
    #                 create_vector_index(pdf_path)

    #         with st.spinner(f"Getting answer from {pdf_file}..."):
    #             try:
    #                 docs = get_relevant_docs(index_path, user_question)
    #                 context = "\n\n".join([doc.page_content for doc in docs])
    #                 answer = generate_response(context, user_question)
    #                 st.write(f"### 📄 Answer from: {pdf_file}")
    #                 st.write(answer)
    #             except Exception as e:
    #                 st.error(f"Error with {pdf_file}: {str(e)}")

def main():
    login()
    if st.session_state.get("logged_in"):
        main_app()

if __name__ == "__main__":
    main()