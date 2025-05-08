from flask import Flask,request, jsonify
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

app = Flask(__name__)

#loading API KEY dan url Database
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
db_url = os.getenv("DATABASE_URL")

#cek .env API dan url Database
if not groq_api_key or not db_url:
    raise EnvironmentError("Please set your GROQ_API_KEY and DATABASE_URL in .env")

#membuat engine untuk export database dan tabel
engine = create_engine(db_url)
query_sql = os.getenv("QUERY_SQL")
df = pd.read_sql_query(query_sql, engine)

#memuat nama model
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

#fungsi mengambil seluruh data dalam tabel
@app.route("/getData",methods=["GET"])
def getData():
    try:
        #mengambil data untuk dibuat dictionary
        data_json = df.to_dict(orient='records')

        #mengubah data dictionary menjadi JSON
        return jsonify(data_json)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#mengambil prompt dari user
@app.route('/chat',methods=['POST'])
def chat():
    try:
        #mengambil prompt dari user dengan request form
        user_input = request.form.get('user_input')

        #memuat model
        chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

        #mengubah dataframe menjadi CSV
        context = df.to_csv(index=False)[:3000]

        #membuat prompt berupa data dan pertanyaan untuk dikirim ke model
        prompt = f"""
        Berikut adalah data hasil query database:
        {context}
        Pertanyaan: {user_input}
        Jawaban:"""

        #memuat jawaban dari model dan menampilkan dengan JSON
        response = chat.invoke([HumanMessage(content=prompt)])
        return jsonify({"answer": response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)