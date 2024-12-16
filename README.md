# Semantic Paper Search

This project is a Streamlit application that allows users to search for semantically similar academic papers based on the title and abstract of a given paper. The application uses a pre-trained SentenceTransformer model to encode the input and corpus papers, and performs semantic search to find the most similar papers.

## Dataset
This application uses a small dataset emnlp paper from 2016-2018 and 2023. The dataset is stored in json format emnlp_data.json. 
The dataset contains the following fields:
- title: the title of the paper
- abstract: the abstract of the paper
- year: the publication year of the paper
- url: the URL of the paper

You could add more papers to the dataset by adding more entries to the json file and retrain again.
```python
    with open('emnlp_data.json', 'r') as f:
        papers = [json.loads(line) for line in f]

    model = SentenceTransformer('allenai-specter')
    corpus_embeddings = model.encode(paper_texts, convert_to_tensor=True, show_progress_bar=True)

    torch.save(corpus_embeddings, 'corpus_embeddings.pt')
```
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
Note: The first run may take a while to download the SentenceTransformer model.

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a PDF paper or manually enter the title and abstract. For example:
    - Title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    - Abstract: "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers."
You could also use the provided example paper under pdf/ directory.

4. Click the "Search" button to find the most similar papers.

## Dependencies

- streamlit
- torch
- sentence-transformers
- PyMuPDF
