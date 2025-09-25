import os
from dotenv import load_dotenv
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import extract
from deepmultilingualpunctuation import PunctuationModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 🔑 Load API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ✅ Global Models
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
punctuation_model = PunctuationModel()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Cache
vectorstore_cache = {}

# -------------------- Transcript Pipeline -------------------- #
def get_transcriptions(video_url: str) -> str:
    try:
        video_id = extract.video_id(video_url)
        transcript = YouTubeTranscriptApi().fetch(video_id)
        if not transcript:
            return "No transcript available for this video."
        raw_text = " ".join(seg.text for seg in transcript)
        return punctuation_model.restore_punctuation(raw_text)
    except Exception as e:
        return f"Transcript Error: {e}"

def get_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)

def build_vectorstore(video_url: str):
    if video_url in vectorstore_cache:
        return vectorstore_cache[video_url]
    transcript = get_transcriptions(video_url)
    chunks = get_chunks(transcript)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore_cache[video_url] = vectorstore
    return vectorstore

# -------------------- RAG Task Factory -------------------- #
def run_rag_task(video_url: str, query: str, prompt_template: str, input_vars=["context"]) -> str:
    retriever = build_vectorstore(video_url).as_retriever(search_kwargs={"k": 8})
    prompt = PromptTemplate(input_variables=input_vars, template=prompt_template)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
    return qa_chain.run(query)

# -------------------- Task Wrappers -------------------- #
def get_summarization(video_url: str) -> str:
    return run_rag_task(
        video_url,
        query="Summarize the transcript",
        prompt_template="Summarize the following transcript into a concise paragraph:\n\n{context}"
    )

def get_keypoints(video_url: str) -> str:
    return run_rag_task(
        video_url,
        query="Extract five key points",
        prompt_template="Extract exactly five key points from the following text:\n\n{context}"
    )

def get_answers(video_url: str, question: str) -> str:
    return run_rag_task(
        video_url,
        query=question,
        input_vars=["context", "question"],
        prompt_template=(
            """You are a helpful assistant. Use ONLY the transcript context to answer.
            If any formula used in the context, please include it in your answer.
            Display the formula clearly using LaTeX format.
            \n\n"""
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

# -------------------- Gradio UI -------------------- #
with gr.Blocks() as demo:
    gr.Markdown("# 🎥 YouTube Transcript RAG with Groq LLM\n\nSummarize, extract key points, or ask questions.")

    video_url = gr.Textbox(label="📺 YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    status = gr.Textbox(label="Status", interactive=False)

    # Preprocess and cache vectorstore when URL is entered
    def preload_vectorstore(video_url):
        build_vectorstore(video_url)
        return "✅ Transcript processed and ready!"

    video_url.change(preload_vectorstore, inputs=video_url, outputs=status)

    with gr.Tab("Summary"):
        summary_btn = gr.Button("Get Summary")
        summary_out = gr.Textbox(label="Summary", lines=10)
        summary_btn.click(get_summarization, inputs=video_url, outputs=summary_out)

    with gr.Tab("Key Points"):
        keypoints_btn = gr.Button("Get Key Points")
        keypoints_out = gr.Textbox(label="Key Points", lines=10)
        keypoints_btn.click(get_keypoints, inputs=video_url, outputs=keypoints_out)

    with gr.Tab("Question Answering"):
        question = gr.Textbox(label="Ask a Question about the Video")
        r_btn = gr.Button("Get Answer")
        r_out = gr.Textbox(label="Answer", lines=10)
        r_btn.click(get_answers, inputs=[video_url, question], outputs=r_out)


# 🚀 Launch
if __name__ == "__main__":
    demo.launch()
