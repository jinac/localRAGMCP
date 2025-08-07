from server import mcp

import pymupdf4llm

import faiss
import numpy as np
import ollama
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter

import json
from pathlib import Path

import re

def get_pdf_txt(txt_filepath):
    txt = pymupdf4llm.to_markdown(
        txt_filepath,
        ignore_images=True,
        ignore_graphics=True,
        page_chunks=True)
    return txt

class FAISSRetriever():
    def __init__(self, db_dir):
        self.db_dir = Path(db_dir)
        self.index = faiss.read_index(str(Path(self.db_dir, "index.faiss")))
        with Path(self.db_dir, "metadata.json").open('r') as f:
            self.metadata = json.load(f)

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=400,
            chunk_overlap=100)
        
        self.embed_model = "mxbai-embed-large"

        self.top_k = 10
        self.d_th = 0.5

    def find_top_chunks(self, query):
        distances, top_k_idxs = self.index.search(query, self.top_k)
        out_idxs = []
        for idx, dist in zip(top_k_idxs, distances):
            if dist <= self.d_th:
                out_idxs.append(idx)

        return(out_idxs)
    
    def retrieve_paper_filepaths(self, idxs):
        filepaths = []
        for idx in idxs:
            filepath = self.metadata[idx]['file']
            filepaths.append(filepath)

        return(filepaths)
    
    def retrieve_paper_context_chunks(self, idxs):
        chunks = []
        for idx in idxs:
            metadata = self.metadata[idx]
            filepath = metadata['file']
            page_idx = metadata['page'] - 1
            chunk_idx = metadata['chunk_id']
            txt = get_pdf_txt(filepath)[page_idx]['text']
            chunk = self.text_splitter.split_text(txt)[chunk_idx]

            chunks.append(chunk)

        return chunks

# x = FAISSRetriever('cvpr2025')
# txt_idxs = [23, 125, 1337]
# y = x.retrieve_paper_filepaths(txt_idxs)
# print(y)
# z = x.retrieve_paper_context_chunks(txt_idxs)
# print(z)

retriever = FAISSRetriever("cvpr2025")

@mcp.tool()
async def search_relevant_papers(query:str) -> str:
    """
    Retrieves relevant papers from document database based on query.

    Args:
        query (str): The search query to find relevant papers.

    Returns:
        str: List of filenames of papers to fit relevance.
    """
    emb = ollama.embed(model=retriever.embed_model, input=query)["embeddings"][0]
    top_chunks = retriever.find_top_chunks(emb)
    top_files = retriever.retrieve_paper_filepaths(top_chunks)
    
    return str(top_files)


@mcp.tool()
async def search_relevant_info(query:str) -> str:
    """
    Retrieves relevant paper contents from document database based on query.

    Args:
        query (str): The search query to find relevant papers.

    Returns:
        str: Concatenated text content from retrieved papers.
    """
    emb = ollama.embed(model=retriever.embed_model, input=query)["embeddings"][0]
    top_chunks = retriever.find_top_chunks(emb)
    top_chunks = retriever.retrieve_paper_context_chunks(top_chunks)

    return str(top_chunks)