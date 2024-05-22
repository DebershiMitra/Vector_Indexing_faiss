import numpy as np
import faiss
import time
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer


embeddings_file_path = 'slack/embeddings'

# Function to split text into chunks
def chunk_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Function to generate embeddings for texts
def generate_embeddings(texts, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings, model

# Function to index vectors using FAISS
def index_vectors(embeddings, index_name="vector_store", update=False):
    if update:
        try:
            index = faiss.read_index(f"{embeddings_file_path}/{index_name}.faiss")
            print("Updating existing index...")
        except FileNotFoundError:
            index = faiss.IndexFlatL2(embeddings.shape[1])
            print("Creating new index...")
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])

    index.add(embeddings)
    faiss.write_index(index, f"{embeddings_file_path}/{index_name}.faiss")

# Function to update the index with new embeddings
def update_index(new_embeddings, index_name="vector_store"):
    index_vectors(new_embeddings, index_name=index_name, update=True)

# Function to handle queries
def handle_query(query, texts, embeddings_file_path, em, similarity_threshold=0.95):
    start_time = time.time()
    model = SentenceTransformer('thenlper/gte-base')
    index = faiss.read_index(f"{embeddings_file_path}")
    query_context_embedding = model.encode([f"{query} Give me a generalized output for the given query"])
    scores, indices = index.search(query_context_embedding, k=3)

    relevant_chunks = []

    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        cosine_similarity = 1.0 - (score / (2 * em.shape[1])) ** 0.5

        if cosine_similarity < similarity_threshold:
            continue

       
        result_text = f"{texts[idx]}".replace("\n", " ")
        relevant_chunks.append(result_text)

    end_time = time.time()
    query_response_time = end_time - start_time

    print(f"\nQuery Response Time: {query_response_time:.4f} seconds")

    return relevant_chunks

# Function to handle new Slack data
def handle_new_slack_data(new_text, model, c_text):

    new_chunked_texts = chunk_text(new_text)
    c_text += new_chunked_texts
    new_embeddings, _ = generate_embeddings(new_chunked_texts, model)
    update_index(new_embeddings, index_name="vector_store")

    return c_text
