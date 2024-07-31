import warnings

# books https://books.ebalbharati.in/
warnings.filterwarnings("ignore")

import os
import logging
from typing import Any, List, Mapping, Optional
import random
from functools import lru_cache
import asyncio
import aiohttp

import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    PyMuPDFLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from flashrank import Ranker, RerankRequest

# Nano (~4MB), blazing fast model & competitive performance (ranking precision).
# ranker = Ranker(model_name="rank-T5-flan", cache_dir="/opt")
ranker = Ranker()


class CustomLLM(LLM):
    models: List[str] = [
        "meta-llama/Llama-3-8b-chat-hf",
        "google/gemma-7b-it",
        "codellama/CodeLlama-13b-Instruct-hf",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ]
    model_index: int = 0
    temperature: float = 0.7
    top_p: float = 0.5

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful teaching assistant."},
            {"role": "user", "content": prompt},
        ]

        ## Custom Rag implmentation ( If you want it then please connect with me )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.models[self.model_index],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


custom_llm = CustomLLM(model_index=0, temperature=0.7, top_p=0.5)


class RAGApplication:
    def __init__(self, persist_directory: str = "db"):
        self.persist_directory = persist_directory
        self.embeddings = FastEmbedEmbeddings(
            model_name="jinaai/jina-embeddings-v2-small-en"
        )
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.cache = {}

    def load_documents(self, file_path: str) -> List[Document]:
        documents = []
        file_path = str(file_path)
        try:
            if file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)  # PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith(".html"):
                loader = UnstructuredHTMLLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return None
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def initialize(self, documents: List[Document], FILE_NAME):
        ## 1) split and chunk the documents ( If you want the code then please connect with me )

        ## 2) Create a vector store ( If you want the code then please connect with me )

        ## 3) Setup the conversation chain as below 
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        print("Vector store created.")
        print("Vector Created in secs : ", time.time() - start)

        print("Setting up QA chain...")
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=custom_llm,
            retriever=self.retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate.from_template(
                    "Please answer the following question using only the provided context. Do not include any information not present in the context. If the context does not contain the answer, state 'Answer not available in the provided context.'Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"
                )
            },
        )
        print("QA chain set up complete.")
        print("QA Chain Created in secs : ", time.time() - start)

    def langchain_to_dict(self, langchain_docs):
        return [
            {"index": i, "text": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(langchain_docs)
        ]

    def dict_to_langchain(self, dict_docs):
        return [
            Document(page_content=doc["text"], metadata=doc["metadata"])
            for doc in dict_docs
        ]

    # Flash reranking , ( If you want the code then please connect with me )
    def flashreranker(self, query, docs):
        pass

    ## Document Retriveal ( If you want the code then please connect with me )
    def retrieve_documents(self, query: str, chat_history: List = []):
        pass
      
    async def achat(self, query: str, chat_history: List = []):
        try:
            print(f"Processing query: {query}")
            top_docs = self.retrieve_documents(query, chat_history)
            chain_input = {
                "question": query,
                "chat_history": chat_history,
                "context": "\n\n".join([doc.page_content for doc in top_docs]),
            }
            print("Generating answer...")
            result = await self.qa_chain.acall(chain_input)
            print("Answer generated.")
            return result["answer"], top_docs
        except Exception as e:
            logger.error(f"Error in chat function: {str(e)}")
            return (
                "I'm sorry, but I encountered an error while processing your request.",
                [],
            )

    def chat(self, query: str, chat_history: List = []):
        return asyncio.run(self.achat(query, chat_history))
