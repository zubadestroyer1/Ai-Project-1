import streamlit as st

import google.generativeai as palm
palm_api_key = st.secrets["PALM_API_KEY"]

def call_palm(prompt: str, palm_api_key: str) -> str:
    palm.configure(api_key=palm_api_key)
    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=prompt,
        temperature=0,
        max_output_tokens=800,
    )

    return completion.result

user_question = st.text_input('Enter a question:', 'Tell me a joke.')
answer = call_palm(prompt=user_question, palm_api_key=palm_api_key)
st.write('Answer', answer)
