import streamlit as st

from api_client import ask_question, check_health

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄")
st.title("📄 RAG Document Assistant")
st.caption("Ask a question about the document collection and get a grounded, cited answer.")

if not check_health():
    st.error("⚠️ Backend is not reachable. Make sure the FastAPI server is running.")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question...")

if question:
    st.session_state.history.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        try:
            result = ask_question(question)
            answer = result["answer"]
            sources = result.get("sources", [])
            content = answer
            if sources:
                content += "\n\n**Sources:** " + ", ".join(sources)
            st.session_state.history.append({"role": "assistant", "content": content})
        except Exception as e:
            st.session_state.history.append(
                {"role": "assistant", "content": f"❌ Something went wrong: {e}"}
            )

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
