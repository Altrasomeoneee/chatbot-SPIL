import streamlit as st
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_csv_agent
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title="LLM SPIL - Ollama")
    st.header("SPIL LLM - Local via Ollama")

    user_csv = st.file_uploader("Upload CSV file", type="csv")
    if user_csv is not None:
        user_question = st.text_input("Tanyakan sesuatu ke data CSV")

        # mengambil model llama3.2
        llm = Ollama(model="llama3.2")

        agent = create_csv_agent(llm, user_csv, verbose=True, allow_dangerous_code=True, max_iterations=20)

        if user_question:
            response = agent.run(user_question)
            st.write(response)

if __name__ == "__main__":
    main()
