# import fitz  # PyMuPDF
import pymupdf
import pymupdf4llm

import faiss
import numpy as np
import ollama
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter

import json
from pathlib import Path

import re

# https://stackoverflow.com/questions/54536539/unicodeencodeerror-utf-8-codec-cant-encode-character-ud83d-in-position-38
def remove_emojis(string):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" # emoticons
        u"\U0001F300-\U0001F5FF" # symbols & pictographs
        u"\U0001F680-\U0001F6FF" # transport & map symbols
        u"\U0001F1E0-\U0001F1FF" # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    
    return emoji_pattern.sub(r'', string)

def get_pdf_txt(txt_filepath):
    txt = pymupdf4llm.to_markdown(
        txt_filepath,
        ignore_images=True,
        ignore_graphics=True,
        page_chunks=True)
    return txt

def generate_metadata(filepath, chunk_idx, chunk):
    md = {
        "file": str(filepath),
        "chunk_id": chunk_idx,
    }
    md.update(chunk['metadata'])
    return md

def build_db(embeddings, metadata, out_dir, index=None):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()


    print(f"Saving Embeddings to {out_dir}")
    
    if index is None:
        index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, str(Path(out_dir, "index.faiss")))

    with Path(out_dir, "metadata.json").open("w") as f:
        f.write(json.dumps(metadata))

    return index

def load_db(db_dir):
    index = faiss.read_index(str(Path(db_dir, "index.faiss")))
    with Path(db_dir, "metadata.json").open('r') as f:
        metadata = json.load(f)

    return index, metadata

def main():
    in_dir = "/mnt/d/cvpr2025/"
    out_dir = "cvpr2025"
    embed_model = "mxbai-embed-large"
    filepaths = [fp for fp in Path(in_dir).glob("*.pdf")]
    # filepaths = filepaths[2337:]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400,
        chunk_overlap=100)

    embeddings = []
    metadata = []
    failures = []
    index = None
    if Path(out_dir).exists():
        index, metadata = load_db(out_dir)
        last_fp = metadata[-1]['file_path']
        restart_idx = filepaths.index(Path(last_fp))+1

        filepaths = filepaths[restart_idx:]

    # print(index)
    # print(last_fp)
    # print(filepaths[22])
    # txt = get_pdf_txt(str(filepaths[22]))
    # print(len(txt))
    # exit()

    for file_idx, fp in enumerate(tqdm(filepaths)):
        try: 
            txt = get_pdf_txt(str(fp))
            for page_idx, page in enumerate(txt):
                txt_chunks = text_splitter.split_text(page['text'])
                for chunk_idx, chunk in enumerate(txt_chunks):
                    metadata.append(generate_metadata(fp, chunk_idx, page))
                    emb = ollama.embed(model=embed_model, input=remove_emojis(chunk))
                    embeddings.append(emb["embeddings"][0])
                    # print(emb)
                    # exit()
            if (file_idx % 100 == 49):
                index = build_db(embeddings, metadata, out_dir, index)
        except Exception as e:
            print(e)
            failures.append(fp)
            continue

    print(failures)

    index = build_db(embeddings, metadata, out_dir, index)


if __name__ == '__main__':
    main()