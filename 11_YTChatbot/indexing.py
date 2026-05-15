from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv; load_dotenv()
from video_ids import video_ids


def get_transcript(video_id):

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(
            video_id=video_id,
            languages=["en"]
        )
        # print(transcript_list)
        transcript = "".join(chunk.text for chunk in transcript_list)

    except TranscriptsDisabled:
        print("No transcript :{")

    return transcript


def get_chunks(transcript):

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    return chunks


def get_embeddings(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
    )

    vector_store = FAISS.from_documents(
        chunks, 
        embedding=embedding_model
    )
    
    return vector_store


def get_vector_store():

    transcript = get_transcript(video_id=video_ids[0])
    chunks = get_chunks(transcript=transcript)
    vector_store = get_embeddings(chunks=chunks)
    
    return vector_store