import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Ensure OpenAI API Key is available
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

# 1. Load and Vectorize the Brand Naming CSV Data
loader = CSVLoader(file_path="brand_naming_logic.csv")
documents = loader.load()
# print(documents[0])
# Extract only text content from documents
text_docs = [doc.page_content for doc in documents]

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Create FAISS vector store
db = FAISS.from_texts(text_docs, embeddings)

# 2. Function for similarity search
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

# 3. Setup Chat Model
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", openai_api_key=openai_api_key)

# Updated Chain using `RunnableLambda`
def generate_response(message):
    best_practice = retrieve_info(message)
    prompt_template = """
    You are an AI-powered brand naming expert. I will describe a brand concept, and you will generate an ideal name based on past best practices.
    Follow these rules:
    1. Ensure the name is unique and memorable.
    2. Use phonetics that are easy to pronounce across different languages.
    3. The name should align with the industry and brand values.
    4. Keep it short and impactful.

    Here is the brand description:
    {message}

    Based on similar naming best practices:
    {best_practice}

    Generate the best possible brand name:
    """
    formatted_prompt = prompt_template.format(message=message, best_practice=best_practice)
    response = llm.invoke(formatted_prompt)
    return response.content

# 4. Build an app with Streamlit
def main():
    st.set_page_config(page_title="Brand Name Generator", page_icon=":rocket:")
    st.header("🚀 Brand Name Generator")
    
    message = st.text_area("Describe your brand (Industry, Values, Style, etc.)")

    if message:
        st.write("Generating a meaningful name for your brand...")

        result = generate_response(message)

        st.success(f"Suggested Brand Name: **{result.strip()}**")

if __name__ == '__main__':
    main()
