
#### Create a templates folder in root and put index.html and upload.html file in it.


#### Components
1) Language : Python (3.10 or higher)
2) Environment : Poetry
3) GEN AI Framework : langchain
4) Vector Store : FAISS
5) Web framework : FASTAPI
6) Reranking : Flashrank
7) Embeddings : Fastembed ("jinaai/jina-embeddings-v2-small-en)
8) LLM : Meta LLAMA 3 8b Chat


[![IMAGE ALT TEXT HERE](Video_Thumbnail.png)](https://www.youtube.com/watch?v=zofEZxCC1Zw)



#### RAG Working
1) Load document
2) Split and chunk
3) Embed , Create Vector store , Intialise the qna chain
4) Fetch similar document to given query
5) Perform reranking
6) Provide the result to llm as context to handle
7) Store quetion and answer in dict


#### To RUN fastapi server
```bash
uvicorn main:app --reload
```

#### Build Container
```bash
docker build -t rag_app_v1 .
```

#### Run Container
```bash
docker run -it -p 8000:8000 rag_app_v1
```
```bash
docker run -d -p 80:5000 -p 443:5000 rag_app_v1 (to redirect http and https traffic directly to application)
```

#### TODO
- Multilingual support (Multilingual embeddings such as cohere or "paraphrase-multilingual-mpnet-base-v2 and llm such as openai or llama3.1)
- Conversation History
- Multimodal Capabilities
- Evaluation and Validation 
- Scalability
- Data Encrption

  
