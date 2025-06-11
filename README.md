```markdown
# Ready Tensor RAG Assistant - Powered by Gemini

🤖 **An intelligent document-powered AI assistant using Google's advanced Gemini AI**

This RAG (Retrieval-Augmented Generation) assistant allows you to chat with your JSON documents using Google's Gemini AI model. Upload your knowledge base and get accurate, context-aware responses powered by state-of-the-art AI.

## ✨ Features

- **🚀 Powered by Google Gemini**: Leverages Gemini-1.5-Flash for fast, accurate responses
- **📚 JSON Knowledge Base**: Upload and chat with your JSON documents
- **💬 Conversational Memory**: Maintains context across the conversation
- **🔍 Intelligent Retrieval**: Uses FAISS vector search for relevant document chunks
- **🌐 Web Interface**: Clean, user-friendly Gradio interface
- **📱 Mobile Responsive**: Works seamlessly on all devices
- **🔒 Secure**: API keys are not stored, only used for the session

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))

### Install Dependencies

```bash
pip install gradio
pip install langchain-google-genai
pip install google-generativeai
pip install faiss-cpu
pip install langchain
```

### Clone and Run

```bash
git clone <your-repo-url>
cd rag-assistant-gemini
python app.py
```

## 🚀 Quick Start

1. **Get Your Gemini API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for use in the application

2. **Prepare Your Knowledge Base**
   - Format your data as a JSON file
   - Supports various JSON structures (objects, arrays, nested data)

3. **Launch the Application**
   ```bash
   python app.py
   ```

4. **Initialize the Assistant**
   - Enter your Gemini API key
   - Upload your JSON knowledge base
   - Click "Initialize Assistant"

5. **Start Chatting**
   - Ask questions about your documents
   - Get intelligent, context-aware responses
   - Enjoy conversational memory across questions

## 📊 Supported JSON Formats

The assistant can process various JSON structures:

### Object Format
```json
{
  "product_1": {
    "name": "Laptop",
    "price": 999,
    "features": ["SSD", "16GB RAM"]
  },
  "product_2": {
    "name": "Phone",
    "price": 699,
    "features": ["5G", "Camera"]
  }
}
```

### Array Format
```json
[
  {
    "id": 1,
    "title": "Document Title",
    "content": "Document content here...",
    "tags": ["tag1", "tag2"]
  },
  {
    "id": 2,
    "title": "Another Document",
    "content": "More content...",
    "tags": ["tag3", "tag4"]
  }
]
```

## 💡 Example Use Cases

- **📖 Documentation Q&A**: Upload your product docs and let users ask questions
- **🛍️ Product Catalogs**: Chat with your product database
- **📊 Data Analysis**: Query your structured data conversationally
- **🎓 Educational Content**: Create interactive learning experiences
- **💼 Business Intelligence**: Get insights from your business data
- **🔍 Research Assistant**: Explore research papers and datasets

## 🎯 Example Queries

- "What information do you have about [specific topic]?"
- "Can you summarize the key points about [subject]?"
- "What are the main features mentioned in the documents?"
- "Help me understand [concept] from the knowledge base"
- "Compare different items in my data"
- "What are the trends or patterns you can identify?"

## ⚙️ Configuration

### Environment Variables
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

### Customization Options

- **Chunk Size**: Modify `chunk_size` in text splitter (default: 1000)
- **Overlap**: Adjust `chunk_overlap` for better context (default: 100)
- **Memory Window**: Change `k` parameter for conversation memory (default: 5)
- **Temperature**: Adjust LLM creativity (default: 0.7)
- **Retrieval Count**: Modify `k` in search_kwargs (default: 4)

## 🔧 Technical Details

### Architecture
- **Frontend**: Gradio web interface
- **LLM**: Google Gemini-1.5-Flash
- **Embeddings**: Google Embedding-001
- **Vector Store**: FAISS
- **Framework**: LangChain
- **Memory**: Conversation Buffer Window

### Performance
- **Fast Responses**: Gemini-1.5-Flash optimized for speed
- **Efficient Retrieval**: FAISS vector similarity search
- **Memory Management**: Sliding window conversation memory
- **Scalable**: Handles large JSON documents efficiently

## 🛡️ Security & Privacy

- **No Data Storage**: Your documents are processed in memory only
- **API Key Security**: Keys are not logged or stored permanently
- **Local Processing**: Vector embeddings created locally
- **Session-Based**: All data cleared when session ends

## 📈 Gemini Advantages

- **🎯 High Accuracy**: State-of-the-art language understanding
- **⚡ Fast Response**: Optimized for speed and efficiency
- **💰 Cost Effective**: Competitive pricing with generous free tier
- **🔄 Long Context**: Handles large documents effectively
- **🌟 Multimodal**: Advanced text processing capabilities

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
git clone <your-repo-url>
cd rag-assistant-gemini
pip install -r requirements.txt
python app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the logs in `rag_assistant.log`
2. Ensure your Gemini API key is valid
3. Verify your JSON file format
4. Open an issue on GitHub

## 🙏 Acknowledgments

- **Google AI** for the powerful Gemini API
- **LangChain** for the excellent RAG framework
- **Gradio** for the beautiful web interface
- **FAISS** for efficient vector search

---

**Ready to get started?** Get your [Gemini API key](https://makersuite.google.com/app/apikey) and start chatting with your documents! 🚀
```
