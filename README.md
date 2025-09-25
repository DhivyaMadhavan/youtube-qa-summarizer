# youtube-qa-summarizer

A Gradio-powered web app that uses Retrieval-Augmented Generation (RAG) to analyze YouTube video transcripts. Powered by Groq's LLaMA 3 and LangChain, this app can summarize videos, extract key points, and answer questions using only the transcript as context.

Features
1. Transcript Extraction: Automatically fetches and punctuates YouTube video transcripts.

2. Chunking & Embedding: Splits transcripts into semantic chunks and embeds them using HuggingFace models.

3. Vector Search: Uses FAISS for fast similarity search across transcript chunks.

4. RAG Pipeline: Combines Groq LLM with LangChain's RetrievalQA for accurate, context-aware responses.

5. Gradio UI: Clean and interactive interface with tabs for summarization, key point extraction, and Q&A.
