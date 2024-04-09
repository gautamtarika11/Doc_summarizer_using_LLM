## Doc_summarizer_using_LLM: Summarize PDFs with Large Language Models

This project empowers you to conquer PDF overload by leveraging the power of Large Language Models (LLMs). It automatically generates summaries of your PDF documents, allowing you to quickly grasp the key points and streamline your workflow.

**Features:**

* **Efficient Summarization:** Gain the gist of lengthy PDFs in seconds, saving you valuable time.
* **Enhanced Information Retrieval:** Easily find relevant information within your PDFs.
* **LLM Powered:** Utilize advanced Large Language Models for accurate summarization.

**Requirements:**

* **streamlit:** For building user-friendly web applications.
* **langchain:** A powerful library for natural language processing pipelines.
* **pypdf2:** Enables working with PDF documents in Python.
* **sentence-transformers:** Facilitates sentence embedding for document similarity.
* **faiss-cpu:** Enables efficient nearest neighbor search for document retrieval (CPU version).
* **openai (optional):** Access OpenAI's LLMs (requires API key).
* **torch, sentencepiece, transformers, accelerate (optional):** Required libraries for certain LLM models.
* **Additional dependencies (optional):** chromadb, tiketoken, fastapi, uvicorn, python-multipart, aiofiles (dependencies for specific functionalities not covered in this core application).

**Getting Started:**

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/your-username/Doc_summarizer_using_LLM.git](https://github.com/gautamtarika11/Doc_summarizer_using_LLM.git)

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **(Optional) Download LLM Model:**
The provided link (https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M/tree/main) points to an LLM model on Hugging Face. You may need to download and place the model files in a specific location within the project (instructions might vary depending on the chosen model).

4. **Run the Application**
    ```bash
    streamlit run Doc_summarizer.py

This will launch the Doc Summarizer application in your web browser, allowing you to upload and summarize your PDF documents.

**Note:**

This README provides a general overview. Specific instructions for downloading and integrating LLM models might vary depending on the chosen model. Refer to the relevant documentation for detailed guidance.
The inclusion of optional dependencies indicates potential functionalities beyond the core summarization functionality. Explore the codebase for further customization possibilities.


