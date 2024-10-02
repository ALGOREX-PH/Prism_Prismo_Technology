import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

# Created by Danielle Bagaforo Meer (Algorex)
# LinkedIn : https://www.linkedin.com/in/algorexph/

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Prism – The Trustworthy Chatbot of Prismo Technology", layout="wide")

with st.sidebar :
    #st.image('images\Prismo_Logo.jpg')
    st.write("Prismo Logo Here")
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "Prism"],
        icons = ['house', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if options == "Home" :
    st.title("Introducing Prism – The Trustworthy Chatbot of Prismo Technology")
    st.write("Prism is the intelligent, trustworthy, and approachable chatbot designed to be the digital face of Prismo Technology. With a deep understanding of the Prismo Accountability Layer and blockchain security, Prism provides users with comprehensive guidance through Prismo’s advanced technological ecosystem. As a key tool for fostering transparency and trust, Prism helps users easily navigate the complexities of blockchain, security features, and the Prismo Technology whitepaper.")
    st.write("## Benefits of Prism for Prismo Technology:")
    st.write("### Enhances User Trust:")
    st.write("Prism simplifies complex blockchain and security concepts, helping users understand how Prismo safeguards their data and ensures transparency. This clear communication fosters a strong sense of trust in the technology, aligning with Prismo’s core values.")
    st.write("### Promotes Transparency:")
    st.write("As the voice of Prismo’s Accountability Layer, Prism gives users detailed insights into how every transaction is logged and verified. This transparency reassures users that their digital interactions are secure, tamper-proof, and fully traceable.")
    st.write("### Efficient Whitepaper Guidance:")
    st.write("Prism serves as a knowledgeable guide to the Prismo Technology whitepaper, breaking down technical details and offering concise summaries. This makes it easy for users to grasp the key benefits and features without needing to sift through complex documents.")
    st.write("### User-Friendly Interaction:")
    st.write("Prism’s conversational approach provides users with an intuitive and friendly experience. It answers queries in a professional yet approachable tone, ensuring users of all technical backgrounds feel supported and informed.")
    st.write("### Reduces Customer Support Load:")
    st.write("By offering instant and accurate responses about Prismo’s technology and whitepaper, Prism significantly reduces the need for extensive customer support, allowing users to find answers quickly without waiting for human assistance.")
    st.write("### Builds a New Era of Trust:")
    st.write("With its focus on trust and security, Prism reinforces Prismo’s mission to create a digital landscape where transparency and integrity are paramount. Prism’s presence ensures users feel confident in every interaction, building a strong foundation of trust in Prismo’s services.")
    st.write("Prism is not just a chatbot but a vital link between users and the cutting-edge technology that Prismo Technology represents. It plays a crucial role in making complex security and blockchain concepts accessible, reinforcing Prismo's dedication to transparency and security.")

elif options == "Prism" :
     dataframed = pd.read_csv('Dataset\Prismo_Knowledgebase.csv')
     print(dataframed)
     documents = dataframed['content'].tolist()
     embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
     embedding_dim = len(embeddings[0])
     embeddings_np = np.array(embeddings).astype('float32')
     index = faiss.IndexFlatL2(embedding_dim)
     index.add(embeddings_np)

     System_Prompt = """
Role:
You are Prism, an authoritative and knowledgeable assistant for Prismo Technology. Your main function is to provide users with clear, detailed, and understandable information about the Prismo Technology whitepaper, including its advanced security features, transparency mechanisms, and the Prismo Accountability Layer.

Instructions:

Greet the user and offer assistance with any questions they have about the whitepaper.
Summarize complex concepts from the whitepaper in simple, user-friendly language.
Provide concise answers to specific questions regarding blockchain, security, transparency, or any other relevant section of the whitepaper.
Highlight key parts of the whitepaper, offering context and insights into why they are important for users.
Clarify technical jargon by providing definitions or breaking them down into layman’s terms.
Ensure users leave each conversation feeling informed and reassured about Prismo Technology’s commitment to security and transparency.

Context:
Prismo Technology is pioneering a new era of trust by using blockchain technology to create secure and transparent digital interactions. The Prismo Accountability Layer is central to this, as it ensures every transaction is tamper-proof and fully traceable. Users rely on Prism to explain how the whitepaper outlines these features and how they contribute to a secure, trustworthy digital environment.

Constraints:

Avoid using overly technical or confusing jargon unless the user specifically asks for technical details.
Always simplify complex concepts and provide analogies when necessary to make information more accessible.
Stay within the boundaries of the Prismo whitepaper content; do not speculate or provide information not covered in the document.
Ensure that all responses are concise, informative, and transparent, aligning with Prismo’s values of trust and integrity.
Examples:

User Question: "What is the Prismo Accountability Layer?"
Prism Response: "The Prismo Accountability Layer is the core technology that ensures every transaction within the Prismo ecosystem is secure and transparent. It uses blockchain to record all interactions in an immutable way, so no one can alter or tamper with the data. This provides full transparency and allows users to verify all actions taken with their data."

User Question: "How does Prismo ensure my data is protected?"
Prism Response: "Prismo uses advanced encryption and blockchain technology to safeguard your data. This means that all data is stored in a way that prevents unauthorized access, and the use of blockchain ensures that every action involving your data is logged and cannot be changed, providing a secure and transparent environment."

User Question: "What are the key takeaways from the whitepaper?"
Prism Response: "The key points of the Prismo Technology whitepaper are: first, the introduction of the Prismo Accountability Layer, which ensures secure and transparent transactions through blockchain; second, the use of advanced security measures like encryption to protect data; and finally, how this technology aims to build a new era of trust in digital interactions."
"""


     def initialize_conversation(prompt):
         if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": System_Prompt})
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            response = chat.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response})

     initialize_conversation(System_Prompt)

     for messages in st.session_state.messages :
         if messages['role'] == 'system' : continue 
         else :
           with st.chat_message(messages["role"]):
                st.markdown(messages["content"])

     if user_message := st.chat_input("Say something"):
        with st.chat_message("user"):
             st.markdown(user_message)
        query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
        query_embedding_np = np.array([query_embedding]).astype('float32')
        _, indices = index.search(query_embedding_np, 5)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = ' '.join(retrieved_docs)
        structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messages + [{"role": "user", "content" : structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        st.session_state.messages.append({"role": "user", "content": user_message})
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
             st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})