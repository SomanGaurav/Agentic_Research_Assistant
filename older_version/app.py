import streamlit as st
import re
import os

# Page config
st.set_page_config(page_title="🔍 Agentic Deep Researcher", layout="wide")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "workflow_phase" not in st.session_state:
    st.session_state.workflow_phase = "RESEARCH"
if "research_context" not in st.session_state:
    st.session_state.research_context = ""

def reset_chat():
    st.session_state.messages = []
    st.session_state.workflow_phase = "RESEARCH"
    st.session_state.research_context = ""

try:
    from agents import run_research_phase, run_implementation_phase
except ImportError as e:
    st.error(f"Error importing agents module: {e}")
    st.stop()

# -------------------------------------------------
# NEW HELPER: Native Image Renderer
# -------------------------------------------------
def render_with_images(text):
    """
    Parses Markdown for image tags, renders the text, and renders images natively via st.image()
    so they actually appear in the Streamlit UI.
    """
    # Regex to find ![Alt Text](File Path)
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    # Split text into chunks: [text, alt1, path1, text, alt2, path2, text...]
    parts = re.split(pattern, text)
    
    # If no images found, just render normal markdown
    if len(parts) == 1:
        st.markdown(text)
        return

    # Loop through the chunks and interleave text and images
    for i in range(0, len(parts), 3):
        # Render the text chunk
        if parts[i].strip():
            st.markdown(parts[i])
            
        # If there is a matching image, render it via st.image
        if i + 1 < len(parts):
            alt_text = parts[i+1]
            img_path = parts[i+2].strip()
            
            # Check if file actually exists before rendering
            if os.path.exists(img_path):
                st.image(img_path, caption=alt_text)
            else:
                st.warning(f"⚠️ Image generated but not found at path: `{img_path}`")


# Header
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("<h2 style='color: #0066cc;'>🔍 Agentic Deep Researcher</h2>", unsafe_allow_html=True)
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Use our new renderer instead of basic st.markdown
        render_with_images(message["content"])

# Dynamic input prompt based on phase
input_placeholder = "Ask a research question..." if st.session_state.workflow_phase == "RESEARCH" else "How would you like to implement or visualize this?"

# Input handling
if prompt := st.chat_input(input_placeholder):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ----------------------------------------
    # PHASE 1: RESEARCH
    # ----------------------------------------
    if st.session_state.workflow_phase == "RESEARCH":
        with st.spinner("Executing Deep Research..."):
            try:
                response = run_research_phase(prompt)
                
                st.session_state.research_context = response
                st.session_state.workflow_phase = "IMPLEMENTATION"
                
                follow_up = "\n\n---\n**✅ Research Complete!** How would you like me to implement this in code or visualize it?"
                response += follow_up

            except Exception as e:
                response = f"Error during research: {str(e)}"
                st.error(f"Debug info: {str(e)}")

    # ----------------------------------------
    # PHASE 2: IMPLEMENTATION
    # ----------------------------------------
    elif st.session_state.workflow_phase == "IMPLEMENTATION":
        with st.spinner("Building Implementation & Visualizations..."):
            try:
                response = run_implementation_phase(
                    research_context=st.session_state.research_context, 
                    user_directive=prompt
                )
                
                st.session_state.workflow_phase = "RESEARCH"
                st.session_state.research_context = ""
                
            except Exception as e:
                response = f"Error during implementation: {str(e)}"
                st.error(f"Debug info: {str(e)}")

    # Add assistant response to UI
    with st.chat_message("assistant"):
        # Use our new renderer here too!
        render_with_images(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})