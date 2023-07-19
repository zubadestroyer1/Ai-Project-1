import streamlit as st
import pandas as pd
import google.generativeai as palm
import numpy as np
import openai
import tensorflow as tf
from PIL import Image

from scipy.spatial.distance import cosine

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# BACKEND
palm_api_key = st.secrets["PALM_API_KEY"]
df = pd.read_csv("question_answer_data_set_list.csv")

st.sidebar.title("Sidebar")
model = st.sidebar.selectbox(
    "Choose which language model do you want to use:",
    ("Palm", "next")
)
domain = st.sidebar.selectbox(
    "Choose which domain you want to search:", ("Text", "Image", "next")
)
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Next item ... ")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

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

def palm_text_embedding(prompt: str, key: str) -> str:
    # API Key
    palm.configure(api_key=key)
    model = "models/embedding-gecko-001"

    return palm.generate_embeddings(model=model, text=prompt)['embedding']


def calculate_sts_palm_score(sentence1: str, sentence2: str, key: str) -> float:
    # Compute sentence embeddings
    embedding1 = palm_text_embedding(sentence1, key) # Flatten the embedding array
    embedding2 = palm_text_embedding(sentence2, key)  # Flatten the embedding array

    # Convert to array
    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    # Calculate cosine similarity between the embeddings
    similarity_score = 1 - cosine(embedding1, embedding2)

    return similarity_score

SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

def call_langchain(prompt: str) -> str:
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True)
    output = agent.run(prompt)

    return output

# reset everything
if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state["number_tokens"] = []
    st.session_state["domain_name"] = []
    counter_placeholder.write(f"Next item ...")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("Enter key words here."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # FRONTEND
    df['similarity'] = df.apply(lambda x: calculate_sts_palm_score(x['question'], prompt, palm_api_key), axis = 1)
    df = df.sort_values(by='similarity', ascending=False)
    context = df['answers'].iloc[0:2]
    top_sim_score = df['similarity'][0]
    #st.dataframe(df)
    
    #Langchain agent for google search
    langchain_search_prompt = f"""
        search information about the key words in or questions in {prompt}.
    """
    
    langchain_response = call_langchain(langchain_search_prompt)
    
    #prompt engineer
    if top_sim_score > 0.85:
        engineered_prompt = f"""
            Based on the context: {context} with similarity score: {top_sim_score},
            answer the following question: {prompt} with correct grammar,
            sentence structure, and substantial details.
        """
    else:
        engineered_prompt = f"""
            Based on the context: {context} with similarity score: {top_sim_score},
            additonally based on this context from the internet: {langchain_response},
            if that similarity score is higher than 0.85 then base most of the answer off of context and not internet context,
            answer the following question: {prompt} with correct grammar,
            sentence structure, and substantial details.
        """
        
    response = call_palm(prompt=engineered_prompt, palm_api_key=palm_api_key)

if domain == "Image":
    st.markdown(
        """
        To learn more about image classification, please refer to this [notebook](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/ref-deeplearning/ex02%20-%20ann%20and%20cnn.ipynb).

        ⚠️⚠️⚠️To interact with the app, you'll need a picture. You can find a sample picture [here](https://github.com/yiqiao-yin/WYN-Vision/tree/main/pics).
    """
    )
    # Load model
    # !!! LOAD CORRECT MODEL ONCE COMPLETE !!!
    new_model = tf.keras.models.load_model("toy_mnist_model.h5")
    if new_model is not None:
        st.success("Load a neural network model successfully.")

    # Load image
    uploaded_file = st.sidebar.file_uploader(
        "Upload your file here...", type=["png", "jpeg", "jpg"]
    )
    if uploaded_file is not None:
        st.image(uploaded_file)

        # Convert to array
        # !!! CHANGE SIE OF IMAGE WITH MODEL ONCE COMPLETE !!!
        w, h = 28, 28
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.write(f"Dimension of the original image: {image.shape}")
        image = np.resize(image, (w, h))
        st.write(f"Dimension of resized image: {image.shape}")

        # Inference
        pred = new_model.predict(image.reshape((1, w, h)))
        label = np.argmax(pred, axis=1)
        st.write(f"Classification Result: {label}")
    else:
        st.warning("Please upload a jpg/png file.")
    
    # Display assistant response in chat message container
    if domain == "Text":
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
 
    
    
