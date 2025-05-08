import streamlit as st
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

def main():
    load_dotenv()
    st.set_page_config(page_title="LLM SPIL - DB via Ollama")
    st.header("SPIL LLM - Database (Local) via Ollama")

    groq_api_key = os.getenv("GROQ_API_KEY")
    # Input your SQL database connection string here
    db_url = os.getenv("DATABASE_URL")  # e.g., "postgresql://user:pass@localhost:5432/mydb"
    if not groq_api_key or not db_url:
        st.error("Please set your GROQ_API_KEY and DATABASE_URL in .env")
        return

    engine = create_engine(db_url)
    user_query = "SELECT * FROM pembelian2"
    user_question = st.text_input("Apa yang ingin kamu tanyakan tentang hasil data?")
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
    if user_query:
        try:
            df = pd.read_sql_query(user_query, engine)
            st.dataframe(df)

            if user_question:
                chat = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name =model_name
                )
                context = df.to_csv(index=False)[:3000]  # Truncate if needed
                prompt = f"""
                Berikut adalah data hasil query database:
                {context}

                Pertanyaan: {user_question}
                Jawaban:"""
                response = chat.invoke([HumanMessage(content=prompt)])
                st.write(response.content)

        except Exception as e:
            st.error(f"Terjadi error: {e}")

if __name__ == "__main__":
    main()
