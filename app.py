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

# 1. Load and Vectorize the Webibee Services CSV Data
loader = CSVLoader(file_path="webibee_services_dataset.csv")
documents = loader.load()

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
def analyze_job_post(job_post):
    matching_services = retrieve_info(job_post)
    prompt_template = """
    You are an AI-powered job analysis assistant. I will provide a job description from Upwork, and you will analyze if our agency can take the job.
    
    Based on the available data about our services, determine:
    - If our services match at least 70% of the job post requirements.
    - If the match is lower than 70%, respond that we should not take the job and explain why.
    - If the match is higher than 70%, highlight the specific services that match and explain why we are a good fit.
    - Suggest minor tweaks in our approach if required.
    
    Here is the job description:
    {job_post}
    
    Based on our available services:
    {matching_services}
    
    Provide an evaluation with a percentage match, reasoning for the decision, and a recommendation.
    """
    
    formatted_prompt = prompt_template.format(job_post=job_post, matching_services=matching_services)
    response = llm.invoke(formatted_prompt)
    return response.content

# 4. Build an app with Streamlit
def main():
    st.set_page_config(page_title="Ask Webibee - Job Match Analysis", page_icon="🦾")
    st.title("Ask Webibee 🦾")
    st.write("Let's see if we can take up this job!")
    
    job_post = st.text_area("Paste the Upwork Job Post Here", height=350)

    if job_post:
        st.write("Analyzing your job requirement...")
        result = analyze_job_post(job_post)

        # print("DEBUG OUTPUT:", result.lower())
        
        # Keywords that indicate a bad fit
        negative_keywords = [
            "does not match",
            "not a good fit",
            "significant mismatch",
            "not suitable",
            "less than 70%",
            "we should not take this job",
            "do not match at least 70% of the requirements",
            "Since our services do not align with the specific requirements of this job"
        ]
        
        # Keywords that indicate a good fit
        positive_keywords = [
            "match more than 70%",
            "recommend taking this job",
            "we are a good fit",
            "we should take this job"
        ]

        if any(keyword in result.lower() for keyword in negative_keywords):
            st.error("🚫 Not a good fit:\n" + result.strip())
        elif any(keyword in result.lower() for keyword in positive_keywords):
            st.success("✅ We can take it!\n" + result.strip())
        else:
            # st.warning("⚠️ Unable to determine fit. Please review manually:\n" + result.strip())
            st.error("🚫 Not a good fit:\n" + result.strip())

if __name__ == '__main__':
    main()
