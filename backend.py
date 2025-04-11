import os
import whisper
import yt_dlp
import tempfile
import replicate
from fastapi import FastAPI, UploadFile, Form
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_milvus import Milvus
from langchain_community.llms import Replicate
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI()
from transformers import AutoModel
from huggingface_hub import login


# Load Embedding Model
embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_path)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

# Set Up Vector Database
db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
vector_db = Milvus(
    embedding_function=embeddings_model,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)

# Load LLM from Replicate
REPLICATE_API_TOKEN = os.getenv("Replicate Token add here")
model_path = "ibm-granite/granite-3.2-8b-instruct"
model = Replicate(model=model_path, replicate_api_token=REPLICATE_API_TOKEN)


# Whisper for Transcription
def download_audio(video_url, output_path="audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,  # yt-dlp may append the format extension
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Ensure correct filename
    if not os.path.exists(output_path) and os.path.exists(output_path + ".mp3"):
        os.rename(output_path + ".mp3", output_path)

    return output_path


def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


@app.post("/process_youtube")
async def process_youtube(video_url: str = Form(...)):
    audio_file = download_audio(video_url)
    transcript = transcribe_audio(audio_file)
    return {"transcript": transcript}


@app.post("/process_document")
async def process_document(file: UploadFile):
    contents = await file.read()
    text = contents.decode("utf-8")

    # Split text and store in vector DB
    text_chunks = text.split("\n\n")  # Simple chunking
    documents = [{"text": chunk, "metadata": {"doc_id": i}} for i, chunk in enumerate(text_chunks)]
    vector_db.add_documents(documents)

    return {"message": f"Processed {len(documents)} document chunks."}


@app.post("/query_rag")
async def query_rag(query: str = Form(...)):
    docs = vector_db.similarity_search(query)
    prompt_template = PromptTemplate.from_template(template="{input}\n\n{context}")

    combine_docs_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt_template,
        document_prompt=PromptTemplate.from_template(template="{page_content}"),
        document_separator="\n\n",
    )
    rag_chain = create_retrieval_chain(
        retriever=vector_db.as_retriever(),
        combine_docs_chain=combine_docs_chain,
    )

    output = rag_chain.invoke({"input": query})
    return {"answer": output["answer"]}
