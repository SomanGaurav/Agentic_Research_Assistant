# import streamlit as st
# import os
# from agentic_crew import research_crew # Ensure this matches your filename

# # 1. Page Configuration
# st.set_page_config(page_title="AI Research Assistant", page_icon="🧬", layout="wide")

# # Ensure the local papers directory exists
# if not os.path.exists("papers"):
#     os.makedirs("papers")

# st.title("🧬 Agentic Academic Research Assistant")
# st.markdown("Navigate arXiv or analyze local papers with an expert multi-agent team.")

# # 2. Sidebar for Configuration and Uploads
# with st.sidebar:
#     st.header("Research Assets")
#     uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])
    
#     if uploaded_file:
#         file_path = os.path.join("papers", uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         st.success(f"File '{uploaded_file.name}' ready for analysis.")
    
#     st.divider()
#     st.info("The system uses Gemini 2.5 and a hierarchical agent process to coordinate tasks.")

# # 3. Main Chat Interface
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # 4. Handling the User Input
# if prompt := st.chat_input("What would you like to research?"):
#     # Display user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # 5. Execute CrewAI
#     with st.chat_message("assistant"):
#         with st.status("Manager Agent Coordinating Crew...", expanded=True) as status:
#             try:
#                 # Prepare Inputs
#                 # We use .get() to handle cases where file_name might not be used
#                 crew_inputs = {
#                     'user_input': prompt,
#                     'query': prompt, # Matches your hypothesis_task placeholder
#                     'file_name': uploaded_file.name if uploaded_file else "None"
#                 }

#                 # Kickoff the Crew
#                 # The Manager Agent decides which tasks to fire based on the inputs
#                 result = research_crew.kickoff(inputs=crew_inputs)
                
#                 status.update(label="Research Complete!", state="complete", expanded=False)
#                 st.markdown(result)
                
#                 # Save to history
#                 st.session_state.messages.append({"role": "assistant", "content": str(result)})

#             except Exception as e:
#                 st.error(f"An error occurred during research: {e}")
#                 status.update(label="Research Failed", state="error")

# # 6. Optional: Data Persistence/Export
# if st.session_state.messages:
#     if st.button("Clear Conversation"):
#         st.session_state.messages = []
#         st.rerun()

import streamlit as st
import os
from agentic_crew import run                          # ← changed
from chart_runner import extract_and_run_chart 
st.set_page_config(page_title="AI Research Assistant", page_icon="🧬", layout="wide")

if not os.path.exists("papers"):
    os.makedirs("papers")

st.title("🧬 Agentic Academic Research Assistant")
st.markdown("Navigate arXiv or analyze local papers with an expert multi-agent team.")

with st.sidebar:
    st.header("Research Assets")
    uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file:
        file_path = os.path.join("papers", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' ready for analysis.")
    
    st.divider()
    st.info("The system uses Gemini 2.5 and a hierarchical agent process to coordinate tasks.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chart"):                             # ← new
            st.image(message["chart"])    

if prompt := st.chat_input("What would you like to research?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Planning and executing research workflow...", expanded=True) as status:
            try:
                result = run(                                          # ← changed
                    user_input=prompt,
                    file_name=uploaded_file.name if uploaded_file else None
                )

                status.update(label="Research Complete!", state="complete", expanded=False)
                st.markdown(result)
                result_str = str(result)
                status.update(label="Research Complete!", state="complete", expanded=False)
                chart_path = extract_and_run_chart(result_str)
                if chart_path:
                    st.image(chart_path, caption="Generated Comparison Chart")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result_str,
                        "chart": chart_path       # save path for history replay
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result_str
                    })
                st.session_state.messages.append({"role": "assistant", "content": str(result)})

            except Exception as e:
                st.error(f"An error occurred: {e}")
                status.update(label="Research Failed", state="error")

if st.session_state.messages:
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()