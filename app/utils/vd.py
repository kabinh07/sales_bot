from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from app.config import QDRANT_URL, EMB_MODEL_NAME

embedding_model = HuggingFaceEmbeddings(model_name=EMB_MODEL_NAME)
QDRANT_CLIENT = QdrantClient(url=QDRANT_URL)
QDRANT_CLIENT.recreate_collection(
    collection_name="chatbot_context",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

VECTOR_STORE = QdrantVectorStore(client=QDRANT_CLIENT, collection_name="chatbot_context", embedding=embedding_model)

BOOTCAMP_DOC = """
AI Mastery Bootcamp
● Duration: 12 weeks
● Price: $499 (special offer: $299)
● Key Benefits:
○ Learn LLMs, Computer Vision, and MLOps
○ Hands-on projects
○ Job placement assistance
○ Certificate upon completion
"""
VECTOR_STORE.add_texts([BOOTCAMP_DOC], metadatas=[{"type": "bootcamp_info"}])
