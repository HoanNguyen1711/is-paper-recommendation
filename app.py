import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
import json
import fitz  # PyMuPDF
import io

# Load the model and embeddings
model = SentenceTransformer('allenai-specter')

# Load precomputed corpus embeddings (after you save them with torch)
corpus_embeddings = torch.load('corpus_embeddings.pt')

# Load the papers
with open('emnlp_data.json', 'r') as f:
    papers = [json.loads(line) for line in f]

# Define the search function
def search(title, abstract):
    query_embedding = model.encode(title + '[SEP]' + abstract, convert_to_tensor=True)
    search_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)[0]
    
    results = []
    for hit in search_hits:
        related_paper = papers[hit['corpus_id']]
        results.append({
            'title': related_paper['title'],
            'abstract': related_paper['abstract'],
            'url': related_paper['url'],
            'year': related_paper['year'],
            'score': hit['score']
        })
    return results

def extract_pdf_info(pdf_file):
    # Read the uploaded file into memory as bytes
    pdf_bytes = pdf_file.getvalue()

    # Use PyMuPDF (fitz) to open the PDF from the byte stream
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")  # Specify stream and filetype explicitly

    title = None
    abstract = None

    # Try extracting title from metadata first
    metadata = doc.metadata
    if metadata.get('title'):
        title = metadata['title']

    # If no metadata title, try extracting from the largest font on the first page
    if not title:
        page = doc[0]  # Use the first page
        blocks = page.get_text("dict")["blocks"]
        
        largest_font_size = 0
        for block in blocks:
            if block['type'] == 0:  # Text block
                for line in block['lines']:
                    for span in line['spans']:
                        if span['size'] > largest_font_size:
                            largest_font_size = span['size']
                            title = span['text'].strip()
        
        # If still no title, take the first non-empty line (fallback)
        if not title:
            for page in doc:
                text = page.get_text("text")
                lines = text.split("\n")
                title = next((line.strip() for line in lines if line.strip()), None)
                if title:
                    break

    # Extract abstract (assuming it's after 'Abstract' in the text)
    if not abstract:
        in_abstract = False
        abstract_lines = []

        for page in doc:
            text = page.get_text("text")
            lines = text.split("\n")
            
            for line in lines:
                line = line.strip()

                if line.lower().startswith("abstract"):
                    in_abstract = True
                    # Skip the line that contains "Abstract" itself
                    continue

                if in_abstract:
                    # Append lines to abstract until we reach an empty line or section
                    if line == "":
                        break
                    abstract_lines.append(line)

            if in_abstract and abstract_lines:
                break  # Exit the loop once the abstract is collected

        # Combine the lines into a single paragraph
        abstract = " ".join(abstract_lines).strip() if abstract_lines else None

    return title, abstract

# Streamlit UI
st.title('Semantic Paper Search')

# File upload
uploaded_file = st.file_uploader("Upload a PDF paper", type="pdf")

title = None
abstract = None

if uploaded_file:
    # Use the uploaded file
    title, abstract = extract_pdf_info(uploaded_file)
    
    # Pre-fill the textboxes with extracted title and abstract, allowing editing
    title = st.text_input('Enter paper title:', value=title or "")
    abstract = st.text_area('Enter paper abstract:', value=abstract or "")

# If the title and abstract are not from the uploaded PDF, allow text input
if not uploaded_file or not (title and abstract):
    title = st.text_input('Enter paper title:')
    abstract = st.text_area('Enter paper abstract:')

if st.button('Search'):
    if title and abstract:
        results = search(title, abstract)
        
        st.subheader('Most Similar Papers')
        for i, result in enumerate(results):
            paper_link = f"[**{i+1}. {result['title']} ({result['year']})**]({result['url']})"
            st.markdown(paper_link, unsafe_allow_html=True)
            st.write(f"**Score:** {result['score']:.2f}")
            st.write(result['abstract'])
            st.write('\n')
    else:
        st.write('Please enter both title and abstract.')
