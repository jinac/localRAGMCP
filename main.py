# import fitz  # PyMuPDF
import pymupdf4llm

import faiss
import numpy as np
import ollama
# from langchain_ollama import OllamaEmbeddings

# from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_community.vectorstores import FAISS
import faiss
import json
# import tiktoken


# def get_chunks(text, chunk_size=300, overlap=30):
#     enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
#     tokens = enc.encode(text)
#     chunks = []
#     for i in range(0, len(tokens), chunk_size - overlap):
#         chunk = tokens[i:i + chunk_size]
#         chunks.append(enc.decode(chunk))
#     return chunks

# chunks = get_chunks(scraped_text)

# embeddings = [
#     openai.embeddings.create(input=[chunk], model="text-embedding-3-small")["data"][0]["embedding"]
#     for chunk in chunks
# ]

# index = faiss.IndexFlatL2(len(embeddings[0]))
# index.add(np.array(embeddings).astype("float32"))

def get_pdf_text(txt_filepath):
    txt = pymupdf4llm.to_markdown(txt_filepath)
    return txt

def main():
    # https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html
    txt_filepath = "/mnt/d/cvpr2025/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.pdf"
    # md_txt = pymupdf4llm.to_markdown(txt_filepath)
    # print(md_txt)
    txt = get_pdf_text(txt_filepath)
    txt1_filepath = "/mnt/d/cvpr2025/Zhuravlev_Denoising_Functional_Maps_Diffusion_Models_for_Shape_Correspondence_CVPR_2025_paper.pdf"
    txt1 = get_pdf_text("/mnt/d/cvpr2025/Zhuravlev_Denoising_Functional_Maps_Diffusion_Models_for_Shape_Correspondence_CVPR_2025_paper.pdf")

    # embeddings = OllamaEmbeddings(
    #     model="mxbai-embed-large",
    # )
    # print(embeddings)

    embeddings = [
        ollama.embed(model="mxbai-embed-large", input=d)["embeddings"][0]
        for d in [txt, txt1]
    ]
    metadata = [
        {"filepath": txt_filepath},
        {"filepath": txt1_filepath}
    ]
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    print(np.array(embeddings).shape)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, "test_index.faiss")

    index_reload = faiss.read_index("test_index.faiss")
    print(index)
    print(index_reload)

    # vectorstore = InMemoryVectorStore.from_texts(
    #     [txt, txt1],
    #     embedding=embeddings
    # )
    # index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))


    # for idx, (id, doc) in enumerate(vectorstore.store.items()):
    #     print(idx, doc['metadata'])

    # retriever = vectorstore.as_retriever()

    # retrieved_documents = retriever.invoke("What document talks about Streaming 3D Occupancy?")


if __name__ == '__main__':
    main()