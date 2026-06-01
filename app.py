import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Smart PDF Q&A",
    page_icon="📘",
    layout="wide"
)

# ==================================================
# BACKGROUND IMAGE + DESIGN
# ==================================================
st.markdown("""
<style>

.stApp{
    background-image:url("https://images.unsplash.com/photo-1516321318423-f06f85e504b3");
    background-size:cover;
    background-position:center;
    background-repeat:no-repeat;
    background-attachment:fixed;
}

.main{
    background:rgba(255,255,255,0.15);
}

h1,h2,h3,label{
    color:white !important;
}

[data-testid="stFileUploader"]{
    background:rgba(255,255,255,0.9);
    padding:15px;
    border-radius:15px;
}

[data-testid="stRadio"]{
    background:rgba(255,255,255,0.9);
    padding:10px;
    border-radius:15px;
}

.stTextInput input{
    background:white;
    border-radius:10px;
}

.stButton button{
    width:100%;
    height:50px;
    border:none;
    border-radius:10px;
    background:#0066ff;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.result-box{
    background:rgba(255,255,255,0.9);
    padding:15px;
    border-radius:15px;
    color:black;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================
st.markdown("""
<h1 style='text-align:center'>
📘 Smart PDF Q&A & Word Explainer
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div class='result-box'>
<b>Features:</b><br>
✅ Upload Any PDF<br>
✅ Ask PDF Questions<br>
✅ Get Simplified Answers<br>
✅ Enter Any Word<br>
✅ Meaning, Uses, Why Used, When Used
</div>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ==================================================
# PDF FUNCTIONS
# ==================================================
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=500):
    return [
        text[i:i+chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def build_index(chunks):

    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(np.array(embeddings))

    return index


def search_pdf(question, chunks, index):

    q_emb = model.encode([question])

    D, I = index.search(
        np.array(q_emb),
        k=3
    )

    return [chunks[i] for i in I[0]]

# ==================================================
# WORD EXPLAINER
# ==================================================
def explain_word(word):

    meaning = f"{word} is an important concept used in education, research and practical applications."

    uses = [
        f"{word} is used in education.",
        f"{word} is used in research.",
        f"{word} is used in projects.",
        f"{word} is used in industry."
    ]

    why = [
        f"{word} helps improve understanding.",
        f"{word} supports learning.",
        f"{word} makes work easier."
    ]

    when = [
        f"Used while studying {word}.",
        f"Used in projects and reports.",
        f"Used in professional work."
    ]

    return meaning, uses, why, when

# ==================================================
# SESSION STATE
# ==================================================
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

# ==================================================
# PDF UPLOAD
# ==================================================
uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    text = extract_text_from_pdf(uploaded_file)

    chunks = split_text(text)

    index = build_index(chunks)

    st.session_state.chunks = chunks
    st.session_state.index = index

    st.success("✅ PDF Uploaded Successfully")

# ==================================================
# MODE
# ==================================================
mode = st.radio(
    "Select Mode",
    ["PDF Q&A", "Word Explainer"]
)

# ==================================================
# PDF Q&A
# ==================================================
if mode == "PDF Q&A":

    question = st.text_input(
        "Ask Question From PDF"
    )

    if st.button("Generate Answer"):

        if st.session_state.chunks is None:

            st.warning(
                "Please upload a PDF first."
            )

        elif question == "":

            st.warning(
                "Please enter a question."
            )

        else:

            results = search_pdf(
                question,
                st.session_state.chunks,
                st.session_state.index
            )

            st.subheader("📌 Answer")

            for r in results:

                st.markdown(
                    f"""
                    <div class='result-box'>
                    {r}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ==================================================
# WORD EXPLAINER
# ==================================================
if mode == "Word Explainer":

    word = st.text_input(
        "Enter Any Word"
    )

    if st.button("Generate Meaning"):

        if word == "":

            st.warning(
                "Please enter a word."
            )

        else:

            meaning, uses, why, when = explain_word(word)

            st.markdown(
                f"""
                <div class='result-box'>
                <h3>Meaning</h3>
                {meaning}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class='result-box'>
                <h3>Uses</h3>
                {'<br>'.join(uses)}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class='result-box'>
                <h3>Why Used</h3>
                {'<br>'.join(why)}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class='result-box'>
                <h3>When Used</h3>
                {'<br>'.join(when)}
                </div>
                """,
                unsafe_allow_html=True
            )
