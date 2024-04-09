"""TEXT SUMMARIZATION Web APP"""

# Importing Packages
import base64
import streamlit as st
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline

# Model and Tokenizer
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float32)


# File Loader & Processing
def file_processing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts


# Language Model Pipeline -> Summarization
def llm_pipeline(filepath, summary_length):
    pipe_summ = pipeline(
        "summarization",
        model=base_model,  # T5ForConditionalGeneration.from_pretrained(checkpoint),
        tokenizer=tokenizer,  # T5Tokenizer.from_pretrained(checkpoint),
        max_length=summary_length,
        min_length=50,
    )
    input = file_processing(filepath)
    result = pipe_summ(input)
    result = result[0]["summary_text"]
    return result


# Streamlit Code
st.set_page_config(layout="wide")


# Display Background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        opacity:0.9;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


add_bg_from_local("Images/lamp_background.jpg")

# Font Style
with open("font.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


# Sidebar content


st.sidebar.image("Images/sidebar.png")
st.sidebar.title("About Project:- ")
st.sidebar.write(
    "This project tackles PDF overload! It utilizes advanced Large Language Models (LLMs) to automatically summarize PDF documents." 
    "Get the gist of lengthy PDFs quickly, improving information retrieval and streamlining your workflow."
)


# Email address and LinkedIn URL
email_address = "gautam.11062001.tarika@gmail.com"
linkedin_url = "https://www.linkedin.com/in/gautam-tarika/"

st.sidebar.write(
    "\n**Email:** <a href='mailto:" + email_address + "' style='color: yellow;'>Write to Me!</a>"
    "\n"
    "\n**Linkedin:** <a href=" + linkedin_url + " style='color: yellow;'>Gautam Tarika</a>"
    , unsafe_allow_html=True
)

# Display pdf of a given file
@st.cache_data
def display(file):
    # Opening file from filepath
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    # Embedding pdf in html
    display_pdf = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" '
        f'type="application/pdf"></iframe>'
    )
    # Displaying File
    st.markdown(display_pdf, unsafe_allow_html=True)


# Main content
st.markdown(
    """
    <style>
    .project-title {
        font-size: 57px;
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .project-title span {
        transition: color 0.2s ease-in-out;
    }
    .project-title:hover span {
        color: #f5fefd; /* Hover color */
    }
    .project-title:hover {
        transform: scale(1.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#Heading
text = "Mini Project"  # Text to be styled
# colored_text = ''.join([
#     '<span style="color: hsl(220, 60%, {}%);">{}</span>'.format(70 - (i * 10 / len(text)), char) for i, char in enumerate(text)
# ])

colored_text = ''.join([

    '<span style="color: hsl(60, 100%, {}%);">{}</span>'.format(70 - (i * 10 / len(text)), char) for i, char in enumerate(text)
])
colored_text_with_malt = '<span style="color: hsl(60, 100%, 70%);">&#x2727;</span> ' + colored_text + ' <span style="color: hsl(60, 100%, 70%);">&#x2727;</span>'
center_aligned_style = '<h1 class="Mini-Project-title" style="text-align: center;">'
centered_content = center_aligned_style + colored_text_with_malt + '</h1>'
st.markdown(centered_content, unsafe_allow_html=True)


#Title
st.markdown(
    '<h2 style="font-size:30px;color: #F5FEFD; text-align: center;">Document Summarization using LLMs</h2>',
    unsafe_allow_html=True,
)


# Your Streamlit app content here...
def main():
    # st.title("Mini Project")
    # st.subheader("Document Summarization using Large Language Models")
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    selected_summary_length = st.slider("SELECT SUMMARY STRENGTH", min_value=50, max_value=2000,value=700)
    with st.expander("NOTE"):
        st.write(
            "This currently accepts PDF documents that contain only text and no images. The aim behind this "
            "was to focus on leveraging pre-trained LLMs and utilise them to extract key information from textual content."
        )
    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns((1, 1))
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File:-")
                display(filepath)
            with col2:
                st.spinner(text="In progress...")
                st.info("Summarised Form:-")
                summary = llm_pipeline(filepath, selected_summary_length)
                st.success(summary, icon="✅")


if __name__ == "__main__":
    main()
