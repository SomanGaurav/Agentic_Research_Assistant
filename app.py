import streamlit as st

# Page config
st.set_page_config(page_title="🔍 Agentic Deep Researcher", layout="wide")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def reset_chat():
    st.session_state.messages = []

# Import research function
try:
    from agents import run_research
except ImportError as e:
    st.error(f"Error importing agents module: {e}")
    st.stop()

# Header
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("<h2 style='color: #0066cc;'>🔍 Agentic Deep Researcher</h2>", unsafe_allow_html=True)
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Researching..."):
        try:
            response = run_research(prompt)
        except Exception as e:
            response = f"Error: {str(e)}"
            st.error(f"Debug info: {str(e)}")

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})