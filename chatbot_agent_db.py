from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from flask import Flask,request, jsonify
from sqlalchemy import create_engine
import pandas as pd
import openai
import os
from dotenv import load_dotenv

app = Flask(__name__)

#loading API KEY dan url Database
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"
db_url = os.getenv('DATABASE_URL')

#cek .env API dan url Database
if not GROQ_API_KEY or not db_url:
    raise EnvironmentError("Please set your GROQ_API_KEY and DATABASE_URL in .env")

#membuat engine untuk export database dan tabel
engine = create_engine(db_url)
query_sql = os.getenv("QUERY_SQL")
df = pd.read_sql_query(query_sql, engine)

#memuat nama model
model_name = "gemma2-9b-it"

if not GROQ_API_KEY or not db_url:
    raise EnvironmentError("Please set your GROQ_API_KEY and DATABASE_URL in .env")

@app.route('/getData',methods=['GET'])
def getData():
    try:
        #mengambil data untuk dibuat dictionary
        data_json = df.to_dict(orient='records')

        #mengubah data dictionary menjadi JSON
        return jsonify({"data": data_json})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=["POST"])
def chat():
    try:
        #mengambil prompt dari user dengan request form
        user_input = request.form.get('user_input')

        #memuat model
        model = ChatOpenAI(
            temperature=0.7,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            openai_api_key=GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1"
        )

        #membuat agent
        agent = create_pandas_dataframe_agent(
            model,
            df,
            verbose=True,
            agent_type="openai-tools",
            allow_dangerous_code=True,
            max_iterations=40
        )

        #mengeluarkan response jawaban
        response = agent.run(user_input)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)