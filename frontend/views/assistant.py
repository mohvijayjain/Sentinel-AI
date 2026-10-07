"""AI assistant: the existing /chat RAG endpoint, nothing else."""

import streamlit as st

import api_client
from components.ui import show_error


EXAMPLES = [
    "What is the current drift status?",
    "Why was the challenger rejected?",
    "वर्तमान ड्रिफ्ट स्थिति क्या है?",
    "Quel est l'état actuel du drift ?",
]


st.markdown("## Sentinel AI Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

st.caption("Try:")
columns = st.columns(len(EXAMPLES))
picked = None
for column, example in zip(columns, EXAMPLES):
    if column.button(example, use_container_width=True):
        picked = example

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about drift, monitoring runs or retraining...") or picked

if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            result = api_client.ask_assistant(question)

        if result.ok:
            answer = result.data.get("answer", "")
            st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})
        elif result.error == api_client.TIMEOUT:
            st.warning("The assistant timed out, try again.")
        elif result.error == api_client.SERVER_ERROR and "500" in result.detail:
            # /chat maps an LLM timeout / failure to 500 "RAG request failed"
            st.warning("The assistant could not answer right now "
                       "(the language model did not respond). Try again.")
        else:
            show_error(result, "Assistant")
