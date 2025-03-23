import streamlit as st
import requests
import pandas as pd
import io
import time
import os

# Try different possible hostnames with fallbacks
FASTAPI_URL = os.environ.get("MODEL_API_URL")
if not FASTAPI_URL:
    # Try common possibilities
    import socket
    try:
        # Try 'model' hostname first
        socket.gethostbyname('model')
        FASTAPI_URL = "http://model:8000"
    except:
        try:
            # Try 'model_api' hostname next
            socket.gethostbyname('model_api')
            FASTAPI_URL = "http://model_api:8000"
        except:
            # Fallback to localhost as last resort
            FASTAPI_URL = "http://127.0.0.1:8000"
            
FASTAPI_ENDPOINT = f"{FASTAPI_URL}/chat"
st.set_page_config(
    page_title="Nakya CoPilot", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "excel_data_text" not in st.session_state:
    st.session_state.excel_data_text = ""

# App header with logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("NakyaLogo.png", width=150)
with col2:
    st.title("CoPilot")

# Create a container for the chat messages
chat_container = st.container()

# Sidebar for uploading an Excel/CSV file
st.sidebar.header("Upload Excel/CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.sidebar.write("Preview of Uploaded Data:")
        st.sidebar.dataframe(df.head())

        # Convert entire DataFrame to a CSV string (or a summarized version)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        st.session_state.excel_data_text = buffer.getvalue()

    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        st.session_state.excel_data_text = ""

def send_message():
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return

    # 1. Add user's message
    st.session_state.conversation.append({"role": "user", "content": user_input})
    st.session_state.user_input = ""  # Clear the input box

    # 2. Send conversation + excel data to FastAPI
    payload = {
        "conversation": st.session_state.conversation,
        "excel_data": st.session_state.excel_data_text or None
    }
    
    # Show a spinner while waiting for the response
    with st.spinner("Thinking..."):
        try:
            response = requests.post(FASTAPI_ENDPOINT, json=payload)
            if response.status_code == 200:
                data = response.json()
                assistant_reply = data["reply"]
                query_results = data["query_results"]
                columns = data["columns"]

                # Add assistant reply
                st.session_state.conversation.append({"role": "assistant", "content": assistant_reply})

                # If any query results were returned, display them
                if query_results and columns:
                    df_results = pd.DataFrame(query_results, columns=columns)
                    # We'll add this as an extra assistant message:
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": f"**Query Results:**\n{df_results.to_markdown(index=False)}"
                    })
            else:
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": f"Server error: {response.status_code}"
                })
        except Exception as e:
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"Request failed: {e}"
            })

# Create a placeholder for the chat messages that will be updated
with chat_container:
    # This ensures the conversation is displayed before the input box
    chat_placeholder = st.empty()
    
    # Create a container for the chat input at the bottom
    input_container = st.container()

# Display conversation in the placeholder
with chat_placeholder.container():
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
    
    # Add some space at the bottom to ensure messages are visible
    st.markdown("<div style='padding: 100px;'></div>", unsafe_allow_html=True)

# Place the input box at the bottom using custom CSS and HTML
st.markdown("""
<style>
.chat-input {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 1rem;
    background-color: white;
    z-index: 1000;
    border-top: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

with input_container:
    # We need to create a container that sits at the bottom
    st.markdown("<div class='chat-input'>", unsafe_allow_html=True)
    
    # Input box + button for user queries
    col1, col2 = st.columns([6, 1])
    with col1:
        st.text_input("Type your question:", key="user_input", on_change=send_message)
    with col2:
        if st.button("Send"):
            send_message()
    
    st.markdown("</div>", unsafe_allow_html=True)

# Auto-scroll to the bottom when new messages are added
if st.session_state.conversation:
    # Use JavaScript to scroll to the bottom
    js = f"""
    <script>
        function scroll_to_bottom() {{
            const chatbox = window.parent.document.querySelector('.main');
            chatbox.scrollTop = chatbox.scrollHeight;
        }}
        scroll_to_bottom();
    </script>
    """
    st.components.v1.html(js, height=0)