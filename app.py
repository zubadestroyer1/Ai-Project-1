import streamlit as st
import pandas as pd
import google.generativeai as palm
import numpy as np
from scipy.spatial.distance import cosine
# BACKEND
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

def calculate_cosine_similarity(sentence1: str, sentence2: str) -> float:
    """
    Calculate the cosine similarity between two sentences.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The cosine similarity between the two sentences, represented as a float value between 0 and 1.
    """
    # Tokenize the sentences into words
    words1 = sentence1.lower().split()
    words2 = sentence2.lower().split()

    # Create a set of unique words from both sentences
    unique_words = set(words1 + words2)

    # Create a frequency vector for each sentence
    freq_vector1 = np.array([words1.count(word) for word in unique_words])
    freq_vector2 = np.array([words2.count(word) for word in unique_words])

    # Calculate the cosine similarity between the frequency vectors
    similarity = 1 - cosine(freq_vector1, freq_vector2)

    return similarity

context = df['answers'].iloc[0:3]
st.dataframe(df)

# FRONTEND
user_question = st.text_input('Enter a question:', 'Ask a question about pink river dolphins.')
df = pd.read_csv("question_answer_data_set_list.csv")
df['similarity'] = df.apply(lambda x: calculate_cosine_similarity(x['question'], user_question), axis = 1)
df = df.sort_values(by='similarity', ascending=False)
#prompt enginee
engineered_prompt = f"""
    Based on the context: {context}, answer the following question: {user_quesstion} with correct grammar and sentence structure.
"""
answer = call_palm(prompt=engineered_prompt, palm_api_key=palm_api_key)
st.write('Answer', answer)
