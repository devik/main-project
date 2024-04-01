import streamlit as st
import os
import openai
import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define a custom document class
class CustomDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Function to initialize Chroma vector store
def initialize_chroma_vector_store(pdf_path):
    try:
        pdf_file = open(pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Calculate average text length per page
        total_text_length = sum(len(page.extract_text()) for page in pdf_reader.pages)
        average_text_length_per_page = total_text_length / len(pdf_reader.pages)

        # Set base and dynamic chunk sizes
        base_chunk_size = 500
        dynamic_chunk_size = int(average_text_length_per_page / base_chunk_size) * base_chunk_size

        # Create CharacterTextSplitter with dynamic chunk size
        text_splitter = CharacterTextSplitter(chunk_size=dynamic_chunk_size)

        # Initialize list to store split text chunks
        split_texts = []

        # Loop through PDF pages, extract text, and split into chunks
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            page_text_chunks = text_splitter.split_text(page_text)
            split_texts.extend(page_text_chunks)

        pdf_file.close()

        # Create list of metadata for each document
        metadata_list = [{"page_number": i} for i in range(len(split_texts))]

        # Create list of CustomDocument instances
        custom_documents = [CustomDocument(page_content=chunk, metadata=metadata) for chunk, metadata in
                            zip(split_texts, metadata_list)]

        # Create OpenAIEmbeddings instance
        embedding_function = OpenAIEmbeddings()

        # Initialize Chroma vector store with custom documents
        db = Chroma.from_documents(custom_documents, embedding_function, persist_directory="./test_db")
        db.persist()

        return db
    except Exception as e:
        st.error(f"Error occurred while initializing Chroma vector store: {e}")
        return None

# Initialize Chroma vector store with PDF document
db = initialize_chroma_vector_store("MANDATORY-DISCLOSURE-2019-.pdf")

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center; font-family: Arial, sans-serif; color: white;'>INFOSEEK</h1>",
    unsafe_allow_html=True
)

# Initialize history list in session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize last_query in session state
if 'last_query' not in st.session_state:
    st.session_state['last_query'] = ""

# Function to get answer for a given query
def get_answer(query):
    try:
        # Perform similarity search
        similar_docs = db.similarity_search(query)

        if similar_docs:
            # Concatenate the user query and content of similar documents
            prompt = f"""Answer this as the administrator of the college. Only answer the question below if you have 100% certainty of the facts, use the context below to answer.
                    Here is some context:
                    {similar_docs[0].page_content}
                    Q: {query}
                    A:"""
            # Use the OpenAI GPT-3.5 API for text generation
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=150  # You can adjust the max_tokens parameter as needed
            )

            # Get the generated answer
            generated_answer = response["choices"][0]["text"].strip(" \n")

            # Display the generated answer
            st.write("Answer:", generated_answer)
            
            # Add query to history
            st.session_state['history'].append(query)
        else:
            st.write("Sorry, no relevant information found.")
    except Exception as e:
        st.error(f"Error occurred while generating answer: {e}")

# Textbox for user query input
user_query = st.text_input("Enter a question:")

# Button to get answer for the query
if st.button("Get Answer"):
    get_answer(user_query)

# Event listener to get answer when user presses Enter key after typing the query
if st.session_state.last_query != user_query:
    st.session_state.last_query = user_query
    get_answer(user_query)

# Display history
st.sidebar.title("History")
for idx, query in enumerate(st.session_state['history']):
    if st.sidebar.button(f"Query {idx+1}: {query}"):
        get_answer(query)
