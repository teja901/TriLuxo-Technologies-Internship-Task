from flask import Flask, request, jsonify
from langchain.document_loaders import WebBaseLoader,UnstructuredHTMLLoader
from langchain_openai import OpenAIEmbeddings
from bs4 import BeautifulSoup
import os
from langchain.vectorstores import FAISS


app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

titles = [] 

def extract_content_from_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    extracted_texts = []

    for doc in docs:
        soup = BeautifulSoup(doc.page_content, 'html.parser')
        text = soup.get_text(separator=" ", strip=True)  
        extracted_texts.append(text)
    return extracted_texts

def create_vector_store():
    url = "https://brainlox.com/courses/category/technical"
    texts = extract_content_from_url(url)

    if not texts:
        print("No content extracted!")
        return None

    embeddings = OpenAIEmbeddings()
    
    vector_db = FAISS.from_texts(texts, embeddings)
    
   
    vector_db.save_local("faiss_index")

    print("Vector store created successfully!")
    return vector_db

def load_vector_store():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings)


@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("message", "")

    if not query:
        return jsonify({"error": "Message is required"}), 400

    vector_db = load_vector_store()

    if not vector_db:
        return jsonify({"error": "Vector database not found."}), 500

    
    results = vector_db.similarity_search(query, k=2)

    response_texts = [result.page_content for result in results] if results else ["No relevant data found."]

    return jsonify({"response": response_texts})

if __name__ == "__main__":
    create_vector_store()
    app.run(debug=True)

