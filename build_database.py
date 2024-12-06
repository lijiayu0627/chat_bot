
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import kagglehub
import os


class FAQDatabase:
    """
        Manages a FAQ database with semantic similarity search
        using SentenceTransformer and FAISS.
    """
    def __init__(self):
        """
            Initializes the database.

            - Loads FAQ data from a Kaggle dataset.
            - Prepares SentenceTransformer for embeddings.
            - Sets up a FAISS index for similarity search.
        """
        path = kagglehub.dataset_download("saadmakhdoom/ecommerce-faq-chatbot-dataset")
        with open(os.path.join(path, 'Ecommerce_FAQ_Chatbot_dataset.json'), 'r') as fp:
            data = json.load(fp)
        self.questions = []
        self.answers = []
        self.data = data['questions']
        for d in self.data:
            self.questions.append(d['question'])
            self.answers.append(d['answer'])
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)

    def build_database(self):
        """
            Encodes questions and builds the FAISS index.
        """
        embeddings = self.model.encode(self.questions, convert_to_numpy=True).astype(np.float32)
        self.faiss_index.add(embeddings)

    def search(self, query, top_k=3):
        """
            Searches for the top_k most similar questions.

            Args:
                query (str): The input query string.
                top_k (int): The number of top matches to retrieve.

            Returns:
                list: Matched questions with answers.
                list: Distances of the matches.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return [f'{self.questions[i]} {self.answers[i]}' for i in indices[0]], distances


class ProductDatabase:
    """
        Manages a product database with semantic similarity search.
    """
    def __init__(self, data_path='sample_data.csv'):
        """
            Initializes the product database.

            - Loads product data from a CSV file.
            - Prepares SentenceTransformer for embeddings.
            - Sets up a FAISS index for similarity search.

            Args:
                data_path (str): Path to the product data CSV file.
        """
        self.data = pd.read_csv(data_path)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.contexts = []

    def build_database(self):
        """
            Encodes product descriptions and builds the FAISS index.
        """
        self.contexts = self.data.apply(
            lambda
                row: f"{row['object']} is in {row['category']} category, priced at ${row['price']}, delivered in {row['delivery date']}.",
            axis=1
        ).tolist()

        embeddings = self.model.encode(self.contexts, convert_to_numpy=True).astype(np.float32)

        self.faiss_index.add(embeddings)

    def search(self, query, top_k=3):
        """
            Searches for the top_k most similar product descriptions.

            Args:
                query (str): The input query string.
                top_k (int): The number of top matches to retrieve.

            Returns:
                list: Matched product descriptions.
                list: Distances of the matches.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return [self.contexts[i] for i in indices[0]], distances
