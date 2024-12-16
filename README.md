# Semantic Paper Search

This project is a Streamlit application that allows users to search for semantically similar academic papers based on the title and abstract of a given paper. The application uses a pre-trained SentenceTransformer model to encode the input and corpus papers, and performs semantic search to find the most similar papers.

## Features

- Upload a PDF paper to automatically extract the title and abstract.
- Manually enter the title and abstract if the PDF is not available.
- Display the top 3 most similar papers with their titles, abstracts, publication years, and similarity scores.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required packages (better to use a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    python3 -m streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a PDF paper or manually enter the title and abstract.

4. Click the "Search" button to find the most similar papers.

## Dependencies

- streamlit
- torch
- sentence-transformers
- PyMuPDF
