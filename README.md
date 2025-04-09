# Research Assistant 📚

An intelligent web-based application that helps you analyze and extract insights from multiple web sources. Built with Streamlit, LangChain, and Groq LLM integration, this tool allows you to research, summarize, and query website content all in one place.


---

## 📑 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [Features In Detail](#-features-in-detail)
  - [URL Processing](#-url-processing)
  - [Question Answering System](#-question-answering-system)
  - [User Interface](#-user-interface)
- [Customization](#-customization)
  - [Modifying the LLM](#modifying-the-llm)
  - [Styling](#styling)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🚀 Features

- 🔍 **URL Analysis**: Analyze content from multiple websites simultaneously  
- 📝 **Automatic Summarization**: Get concise summaries of each processed URL  
- 💬 **Question Answering**: Ask questions about processed content with AI-powered responses  
- 🎯 **Source Attribution**: See which sources contributed to each answer  
- 🔎 **Selective Querying**: Query specific URLs from your collection  

---

## 🛠️ How It Works

1. **Input URLs**: Enter website URLs you want to analyze  
2. **Process Content**: The application fetches, processes, and embeds the content  
3. **Ask Questions**: Query the processed content in natural language  
4. **Get Answers**: Receive AI-generated responses with source references  

---

## 🧰 Installation

### Prerequisites

- Python 3.8+
- Groq API key

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-groq
langchain-community
langchain-huggingface
tiktoken
faiss-cpu
unstructured
requests
python-dotenv
```

---

## 💻 Usage

1. Run the application:

```bash
streamlit run main.py
```

2. Open your browser at `http://localhost:8501`

3. Add URLs in the sidebar and click **"Learn from URLs"**

4. Once processing is complete, use any of the tabs to:
   - Ask questions about all processed content
   - View summaries of each URL
   - Query specific URLs from your collection

---

## 🔬 Features In Detail

### 🌐 URL Processing

The application intelligently processes web content by:

- Fetching content with proper headers  
- Splitting into manageable chunks for processing  
- Creating embeddings for semantic search  
- Automatically generating summaries  

### 🤖 Question Answering System

- Uses FAISS vector database for efficient similarity search  
- Leverages Groq's implementation of Llama-4 to generate accurate responses  
- Provides source attribution to maintain transparency  
- Handles token limitations gracefully  

### 🎨 User Interface

- Modern, responsive design with intuitive layout  
- Real-time status indicators for URL accessibility  
- Interactive tabs for different functionalities  
- Progress indicators for background processing  

---

## 🧩 Customization

### Modifying the LLM

You can change the language model by editing the LLM initialization code:

```python
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # Change to desired model
    temperature=0.9,  # Adjust for creativity vs determinism
    max_tokens=500    # Adjust output length
)
```

### Styling

The application includes custom CSS that can be modified in the `st.markdown()` section at the beginning of the script. Adjust colors, dimensions, and other properties to match your preferred style.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository  
2. Create your feature branch:

```bash
git checkout -b feature/amazing-feature
```

3. Commit your changes:

```bash
git commit -m 'Add some amazing feature'
```

4. Push to the branch:

```bash
git push origin feature/amazing-feature
```

5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web application framework  
- [LangChain](https://python.langchain.com/docs/get_started/introduction) for the document processing pipeline  
- [Groq](https://groq.com/) for the LLM inference API  
- [HuggingFace](https://huggingface.co/) for the embeddings model  

---

