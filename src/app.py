import time
import sys
import streamlit as st

sys.path.append("src")

from rag import run_rag
from evaluate import evaluate_grounding, evaluate_retrieval

st.set_page_config(page_title="Azure RAG Evaluation Platform")

st.title("Enterprise RAG Evaluation & Observability Platform")

question = st.text_input("Ask a question about enterprise documents")

if st.button("Submit"):
    if not question:
        st.warning("Please enter a question.")
    else:
        start = time.time()

        result = run_rag(question)
        grounding = evaluate_grounding(result["question"], result["retrieved_docs"], result["answer"])
        retrieval = evaluate_retrieval(result["question"], result["retrieved_docs"])

        latency = round(time.time() - start, 2)

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Retrieved Documents")
        for doc in result["retrieved_docs"]:
            st.markdown(f"### {doc['source']}")
            st.write(doc["content"])

        st.subheader("Evaluation")
        st.write("Grounding Evaluation")
        st.code(grounding, language="json")

        st.write("Retrieval Evaluation")
        st.code(retrieval, language="json")

        st.subheader("Metrics")
        st.metric("Latency", f"{latency}s")
        st.metric("Token Usage", result["tokens"])
