import os
import streamlit as st
import pickle
import logging
import requests
import tiktoken
from langchain_groq import ChatGroq 
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Primary colors */
    :root {
        --primary-color: #4A6FE3;
        --secondary-color: #6D8EF7;
        --background-color: #f8f9fa;
        --card-bg-color: #ffffff;
        --text-color: #333333;
        --success-color: #28a745;
        --error-color: #dc3545;
        --warning-color: #ffc107;
    }
    
    /* General styling */
    .main {
        background-color: var(--background-color);
        padding: 1rem;
    }
    
    h1, h2, h3 {
        color: var(--primary-color);
        font-weight: 700 !important;
    }
    
    /* Card style */
    .card {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* URL input fields */
    div[data-testid="stTextInput"] {
        background-color: var(--card-bg-color);
        border-radius: 5px;
        padding: 0.25rem;
        margin-bottom: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 5px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 5px 5px 0 0;
        padding: 10px 16px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--card-bg-color) !important;
        color: var(--primary-color) !important;
        border-bottom: 2px solid var(--primary-color) !important;
    }
    
    /* Progress bar */
    div.stProgress > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* Footer */
    footer {
        visibility: hidden;
    }
    
    /* Status indicators */
    .success-status {
        color: var(--success-color);
        font-size: 20px;
    }
    
    .error-status {
        color: var(--error-color);
        font-size: 20px;
    }
    
    /* Query results */
    .answer-box {
        background-color: var(--card-bg-color);
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .source-box {
        background-color: var(--background-color);
        border-left: 2px solid var(--secondary-color);
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.9rem;
    }
    
    /* Responsive layout */
    @media (max-width: 768px) {
        .card {
            padding: 1rem;
        }
        
        .sidebar .sidebar-content {
            padding: 0.5rem !important;
        }
    }
    
    /* Logo and branding */
    .logo-title {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }
    
    .logo-emoji {
        font-size: 2.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--card-bg-color);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: var(--primary-color);
    }
</style>
""", unsafe_allow_html=True)

# Header with logo
st.markdown("""
<div class="logo-title">
    <div class="logo-emoji">📚</div>
    <h1>Research Assistant</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <p>Welcome to your intelligent Research Assistant! Add URLs to learn from, then ask questions about their content.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.markdown("<h3>📌 URL Sources</h3>", unsafe_allow_html=True)
    st.markdown("<p>Add the URLs you want to research:</p>", unsafe_allow_html=True)

# Token counting function using tiktoken
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding
        num_tokens = len(encoding.encode(string))
        return num_tokens
    except Exception:
        # Fallback: estimate tokens as words/0.75 (rough approximation)
        return len(string.split()) // 0.75

# URL validation function
def is_valid_url(url):
    """Check if the URL is valid."""
    if not url:
        return False
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# URL accessibility check function
def check_url_accessibility(url):
    """Check if the URL is accessible."""
    if not is_valid_url(url):
        return False, "Invalid URL format"
    
    try:
        headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"
        }
        response = requests.head(url, headers=headers, timeout=5)
        if response.status_code >= 400:
            return False, f"Server error: HTTP {response.status_code}"
        return True, "OK"
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except requests.exceptions.SSLError:
        return False, "SSL certificate error"
    except requests.exceptions.ConnectionError:
        return False, "Connection error - site might be down"
    except Exception as e:
        return False, f"Error: {str(e)}"

# Initialize session state variables
if 'url_status' not in st.session_state:
    st.session_state.url_status = {}
    
if 'url_count' not in st.session_state:
    st.session_state.url_count = 3  # Start with 3 URL fields

if 'url_data' not in st.session_state:
    st.session_state.url_data = {}  # Store processed data by URL
    
if 'url_processed' not in st.session_state:
    st.session_state.url_processed = set()  # Track which URLs have been processed

if 'url_documents' not in st.session_state:
    st.session_state.url_documents = {}  # Store the document objects for each URL

# Function to add URL field
def add_url_field():
    st.session_state.url_count += 1

# Function to remove URL field
def remove_url_field():
    if st.session_state.url_count > 1:  # Always keep at least one field
        st.session_state.url_count -= 1

# Function to summarize content
def summarize_url_content(url_key, documents, llm):
    # Create a summary prompt with context about the URL
    summary_prompt = f"""
    Please provide a comprehensive summary of the content from this URL: {url_key}
    
    Include the main topics covered, key points, and important findings.
    Structure your summary in bullet points for clarity.
    """
    
    try:
        summary_result = llm.invoke(summary_prompt)
        return summary_result.content
    except Exception as e:
        logging.error(f"Error summarizing URL {url_key}: {str(e)}")
        return f"Error generating summary: {str(e)}"

# Add buttons to manage URL fields in sidebar
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        st.button("➕ Add URL", on_click=add_url_field, use_container_width=True)
    with col2:
        st.button("➖ Remove URL", on_click=remove_url_field, use_container_width=True)

# URL input section with status indicators
urls = []
url_dict = {}  # Dictionary to map URL index to actual URL for later reference

with st.sidebar:
    for i in range(st.session_state.url_count):
        with st.container():
            st.markdown(f"<p style='margin-bottom:5px;font-weight:500;'>URL {i+1}</p>", unsafe_allow_html=True)
            col1, col2 = st.columns([5, 1])
            
            with col1:
                url = st.text_input("", key=f"url_{i}", placeholder="Enter URL here...", label_visibility="collapsed")
            
            if url:
                if url not in st.session_state.url_status:
                    # Only check if the URL has changed or is new
                    is_accessible, message = check_url_accessibility(url)
                    st.session_state.url_status[url] = (is_accessible, message)
                
                is_accessible, message = st.session_state.url_status[url]
                
                with col2:
                    if is_accessible:
                        st.markdown("<div class='success-status'>✓</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='error-status'>✗</div>", unsafe_allow_html=True)
                        st.caption(f"Error: {message}")
            
            if url and is_valid_url(url):
                urls.append(url)
                url_dict[f"URL {i+1}"] = url
    
    st.markdown("<hr>", unsafe_allow_html=True)
    process_url_clicked = st.button("🧠 Learn from URLs", use_container_width=True)

file_path = "faiss_store.pkl"
main_placeholder = st.empty()

# Try to get API key from environment or secrets
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key and hasattr(st, "secrets"):
    groq_api_key = st.secrets.get("GROQ_API_KEY")

# Configure LLM
try:
    llm = ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.9,
        max_tokens=500
    )
except Exception as e:
    st.error(f"Error initializing LLM: {str(e)}")
    st.stop()

# Process URLs when button is clicked
if process_url_clicked:
    if not urls:
        with st.sidebar:
            st.error("Please enter at least one valid URL")
    else:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            try:
                # Create a list of accessible URLs only
                accessible_urls = [url for url in urls if st.session_state.url_status.get(url, (False, ""))[0]]
                
                if not accessible_urls:
                    st.error("None of the provided URLs are accessible. Please check the URLs and try again.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()
                
                # Load data with error handling
                status_placeholder.markdown("<p>📥 Loading data from URLs...</p>", unsafe_allow_html=True)
                progress_bar.progress(10)
                
                # Process each URL separately to allow for individual selection later
                for url in accessible_urls:
                    try:
                        # Skip if already processed
                        if url in st.session_state.url_processed:
                            continue
                            
                        loader = UnstructuredURLLoader(
                            urls=[url],
                            headers={'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"},
                            timeout=30
                        )
                        data = loader.load()
                        
                        if not data:
                            status_placeholder.warning(f"Could not extract content from URL: {url}")
                            continue
                        
                        # Process this specific URL data
                        text_splitter = RecursiveCharacterTextSplitter(
                            separators=['\n\n', '\n', '.', ','],
                            chunk_size=800,
                            chunk_overlap=50
                        )
                        docs = text_splitter.split_documents(data)
                        
                        # Second pass to ensure each chunk is below token limit
                        final_docs = []
                        for doc in docs:
                            content = doc.page_content
                            token_count = num_tokens_from_string(content)
                            
                            if token_count > 1020:
                                smaller_splitter = RecursiveCharacterTextSplitter(
                                    separators=['\n', '.', ',', ' '],
                                    chunk_size=400,
                                    chunk_overlap=20
                                )
                                smaller_docs = smaller_splitter.split_text(content)
                                
                                for smaller_chunk in smaller_docs:
                                    doc_copy = doc.copy()
                                    doc_copy.page_content = smaller_chunk
                                    final_docs.append(doc_copy)
                            else:
                                final_docs.append(doc)
                        
                        # Store the documents for this URL
                        st.session_state.url_documents[url] = final_docs
                        
                        # Create embeddings for this URL
                        status_placeholder.markdown(f"<p>🔄 Processing URL: {url}</p>", unsafe_allow_html=True)
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        vectorstore = FAISS.from_documents(final_docs, embeddings)
                        
                        # Store in session state
                        st.session_state.url_data[url] = vectorstore
                        st.session_state.url_processed.add(url)
                        
                        # Generate and store summary for this URL
                        status_placeholder.markdown(f"<p>📋 Generating summary for: {url}</p>", unsafe_allow_html=True)
                        if final_docs:
                            combined_text = "\n\n".join([doc.page_content for doc in final_docs[:5]])  # Limit to first 5 chunks for summary
                            summary = summarize_url_content(url, combined_text, llm)
                            st.session_state.url_data[f"{url}_summary"] = summary
                        
                    except Exception as e:
                        st.warning(f"Error processing URL: {url}. Error: {str(e)}")
                        logging.error(f"Error processing URL {url}: {str(e)}")
                
                # Also create a combined vectorstore for all URLs
                status_placeholder.markdown("<p>🔄 Creating combined knowledge base...</p>", unsafe_allow_html=True)
                all_docs = []
                for url in st.session_state.url_processed:
                    docs = st.session_state.url_documents.get(url, [])
                    all_docs.extend(docs)
                
                if all_docs:
                    # Create combined embeddings
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    combined_vectorstore = FAISS.from_documents(all_docs, embeddings)
                    
                    # Save combined vectorstore
                    with open(file_path, "wb") as f:
                        pickle.dump(combined_vectorstore, f)
                    
                progress_bar.progress(100)
                status_placeholder.markdown("<p>✅ Processing complete! You can now ask questions about the content.</p>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logging.error(f"Unexpected error: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)

# Create tabs for different functionality
tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📑 URL Summaries", "🔍 Specific URL Query"])

# Tab 1: Standard Q&A functionality
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Ask about all processed content</h3>", unsafe_allow_html=True)
    st.markdown("<p>Ask any question about the content of all the URLs you've added:</p>", unsafe_allow_html=True)
    
    query = st.text_input("", key="query_all", placeholder="Example: What are the main topics discussed in these articles?", label_visibility="collapsed")

    if query:
        try:
            if not os.path.exists(file_path) or not st.session_state.url_processed:
                st.error("Please process URLs first before asking questions.")
            else:
                with st.spinner("🔍 Searching for answers..."):
                    with open(file_path, "rb") as f:
                        try:
                            vectorstore = pickle.load(f)
                            
                            # Set up the QA chain with a try block
                            try:
                                chain = RetrievalQAWithSourcesChain.from_llm(
                                    llm=llm, 
                                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                                )
                                
                                # Process the query with token monitoring
                                result = chain.invoke({"question": query})
                                
                                # Display answer
                                st.markdown("<h4>Answer</h4>", unsafe_allow_html=True)
                                st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)
                                
                                # Display sources, if available
                                sources = result.get("sources", "")
                                if sources:
                                    st.markdown("<h4>Sources</h4>", unsafe_allow_html=True)
                                    sources_list = sources.split("\n")  # Split the sources by newline
                                    for source in sources_list:
                                        if source.strip():  # Only show non-empty sources
                                            st.markdown(f"<div class='source-box'>{source}</div>", unsafe_allow_html=True)
                                            
                            except Exception as e:
                                error_msg = str(e)
                                if "sequence length is longer" in error_msg:
                                    st.error("The retrieved content exceeds the model's token limit. Try a more specific question.")
                                    logging.error(f"Token limit exceeded: {error_msg}")
                                else:
                                    st.error(f"Error processing query: {error_msg}")
                                    logging.error(f"Query processing error: {error_msg}")
                                    
                        except Exception as e:
                            st.error(f"Error loading vector store: {str(e)}")
                            logging.error(f"Vector store loading error: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logging.error(f"Unexpected error in querying: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: URL Summaries
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>URL Content Summaries</h3>", unsafe_allow_html=True)
    
    if st.session_state.url_processed:
        st.markdown("<p>Here are summaries of each URL you've processed:</p>", unsafe_allow_html=True)
        
        for url in st.session_state.url_processed:
            summary = st.session_state.url_data.get(f"{url}_summary")
            if summary:
                with st.expander(f"Summary for {url}"):
                    st.markdown(summary)
            else:
                with st.expander(f"Summary for {url}"):
                    st.markdown("No summary available. The URL may not have been fully processed.")
    else:
        st.info("No URLs have been processed yet. Please add URLs and click 'Learn from URLs'.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Query specific URLs
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Query Specific URLs</h3>", unsafe_allow_html=True)
    
    if not st.session_state.url_processed:
        st.info("No URLs have been processed yet. Please add URLs and click 'Learn from URLs'.")
    else:
        st.markdown("<p>Select specific URLs to query about:</p>", unsafe_allow_html=True)
        
        # Create a multi-select for URL selection
        # Use the URL itself as the option but display it with its index
        url_labels = {url: f"URL {i+1}: {url}" for i, url in enumerate(st.session_state.url_processed)}
        selected_urls = st.multiselect(
            "",
            options=list(st.session_state.url_processed),
            format_func=lambda x: url_labels.get(x, x),
            label_visibility="collapsed"
        )
        
        if selected_urls:
            specific_query = st.text_input("", key="specific_query", placeholder=f"Ask about the {len(selected_urls)} selected URL(s)...", label_visibility="collapsed")
            
            if specific_query:
                with st.spinner("🔍 Searching in selected URLs..."):
                    try:
                        # Combine the selected URL documents for this query
                        selected_docs = []
                        for url in selected_urls:
                            docs = st.session_state.url_documents.get(url, [])
                            selected_docs.extend(docs)
                        
                        if selected_docs:
                            # Create temporary vectorstore from selected documents
                            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                            temp_vectorstore = FAISS.from_documents(selected_docs, embeddings)
                            
                            # Set up QA chain
                            chain = RetrievalQAWithSourcesChain.from_llm(
                                llm=llm, 
                                retriever=temp_vectorstore.as_retriever(search_kwargs={"k": 3})
                            )
                            
                            # Process the query
                            result = chain.invoke({"question": specific_query})
                            
                            # Display answer
                            st.markdown("<h4>Answer (from selected URLs)</h4>", unsafe_allow_html=True)
                            st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)
                            
                            # Display sources
                            sources = result.get("sources", "")
                            if sources:
                                st.markdown("<h4>Sources</h4>", unsafe_allow_html=True)
                                sources_list = sources.split("\n")
                                for source in sources_list:
                                    if source.strip():
                                        st.markdown(f"<div class='source-box'>{source}</div>", unsafe_allow_html=True)
                        else:
                            st.error("No data available for the selected URLs.")
                            
                    except Exception as e:
                        st.error(f"Error processing specific URL query: {str(e)}")
                        logging.error(f"Specific URL query error: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Add a footer with usage instructions in a clean card format
st.markdown("---")
st.markdown("""
<div class="card">
    <h3>How to Use This Tool</h3>
    <ol>
        <li><strong>Add URLs</strong>: Enter URLs in the sidebar fields to analyze</li>
        <li><strong>Process Content</strong>: Click "Learn from URLs" to analyze the content</li>
        <li><strong>Ask Questions</strong>: Use the tabs to ask questions about all content, view summaries, or query specific URLs</li>
    </ol>
    <p>This tool uses AI to analyze and answer questions about web content. All processing happens in your session and is not stored permanently.</p>
</div>
""", unsafe_allow_html=True)

# Hide the "Made with Streamlit" footer
st.markdown('<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>', unsafe_allow_html=True)