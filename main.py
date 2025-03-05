import PyPDF2
import pinecone
from pinecone import Pinecone, ServerlessSpec
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import streamlit as st


#Load enviroment variables
load_dotenv()

#index_host = os.getenv("PINECONE_INDEX_HOST")
index_host = st.secrets["PINECONE_INDEX_HOST"]
#pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
#openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize Pinecone
pc = Pinecone(api_key = pinecone_api_key)

#function to extract all embeddings
def get_embeddings(index_host):
    vec_ids = []
    for i in range(1, 95):
        vec_ids.append(f"chunk_{i}")

    index = pc.Index(host=index_host)
    embeddings = []
    response = index.fetch(ids=vec_ids)

    for id in vec_ids:
        if id in response.vectors:
            embeddings.append(response.vectors[id].values)

    index.close()

    #return the embeddings
    return embeddings
    
#Function to chunk text
def chunking(text, chunk_size):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks

#function to calculate euclidean distance
def euclidean(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
    return distance

#Function to embed a single query
def embed_query(query):
    client = OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    
    embedding = response.data[0].embedding
    return embedding

#Function to get the top 3 best vectors
def knn(query, embeddings, k):
    #embed the intital query
    query_embedding = embed_query(query)

    distances = []

    i = 1
    for embedding in embeddings:
        distance = euclidean(query_embedding, embedding)
        distances.append((i, distance))

        i += 1

    #Sort the distances
    distances.sort(key=lambda x: x[1])

    #Get the top k distances. In our case, we will use k = 3, but it can be altered
    top_k = []
    for i in range(k):
        top_k.append(distances[i][0])

    return top_k

#Function to get the context
def get_context(top_k, chunks):
    context = ""
    for i in top_k:
        context += chunks[i]

    return context

def main():
    #Open the pdf file. Same as setup.py
    with open('surface-hub-user-guide-en-us.pdf', 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""

        for page in range(len(reader.pages)):
            cur_page = reader.pages[page]
            text += cur_page.extract_text()

    # Chunk the text. Same as setup.py
    chunks = chunking(text, 1000)

    # Initialize OpenAI API
    client = OpenAI(api_key=openai_api_key)

    #Get all the embeddings
    embeddings = get_embeddings(index_host)

    query = st.text_input("Enter your question here: ")

    if query:
        top_k = knn(query, embeddings, 3)
        context = get_context(top_k, chunks)
        
        my_input = f" Please answer this question: {query} using this info as context {context}"
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": my_input
                    }
                ]
            )

        #Display the completion
        st.subheader("Answer")
        st.write(completion.choices[0].message.content)

        st.subheader("Context")
        st.write(context)

        #Like or dislike the answer
        rating = st.radio("Did this answer your question?", ("Like", "Dislike"))
        if rating:
            st.write(f"You rated this answer: {rating}")
            #In the future, these, along with the query, context, and actual answer, will be stored in a database for future training

if __name__ == "__main__":
    main()

