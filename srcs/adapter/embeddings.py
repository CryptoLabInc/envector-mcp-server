from typing import List, Union

import numpy as np


class EmbeddingAdapter:
    """
    General Adapter for various embedding SDK interactions.
    """
    def __init__(self, mode: str, model_name: str) -> None:
        self.mode = mode
        self.model_name = model_name

        if mode in ["sbert", "sentence_transformer"]:
            self.adapter = SBERTSDKAdapter(model_name)
        elif mode in ["huggingface", "hf"]:
            self.adapter = HuggingFaceSDKAdapter(model_name)
        elif mode == "openai":
            self.adapter = OpenAISDKAdapter(model_name)
        else:
            raise ValueError(f"Unsupported embedding mode: {mode}")

    def get_embedding(self, texts: List[str]) -> np.ndarray:
        return self.adapter.get_embedding(texts)


class SBERTSDKAdapter:
    """
    Adapter for SBERT (Sentence Transformer) SDK interactions.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """
        Initializes the SBERTSDKAdapter with the provided model name.

        Args:
            model_name (str): The name of the Sentence Transformer model to use.
        """

        from sentence_transformers import SentenceTransformer
        import torch

        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        print(f"SBERT model '{model_name}' loaded.")

    def get_embedding(self, texts: List[str]) -> Union[List[float], List[List[float]]]:
        """
        Retrieves the embedding for the given text using Sentence Transformer SDK.

        Args:
            text (str): The input text to get the embedding vector.

        Returns:
            List[float]: The embedding vector for the input text.
        """
        # text embeddings
        embeddings = self.model.encode(texts)

        # l2 normalize
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize for IP

        assert embeddings.shape[0] == len(texts)

        return embeddings.tolist()


class HuggingFaceSDKAdapter(EmbeddingAdapter):
    """
    Adapter for HuggingFace SDK interactions.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = None) -> None:
        """
        Initializes the HuggingFaceSDKAdapter with the provided model name and cache directory.

        Args:
            model_name (str): The name of the HuggingFace model to use.
            cache_dir (str): The directory to cache the model.
        """

        import torch
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        print(f"HuggingFace model '{model_name}' loaded.")

    def get_embedding(self, texts: List[str]) -> Union[List[float], List[List[float]]]:
        """
        Retrieves the embedding for the given text using HuggingFace SDK.

        Args:
            text (str): The input text to get the embedding vector.

        Returns:
            List[float]: The embedding vector for the input text.
        """
        import torch

        for text in texts:
            # Tokenize sentences
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)

        # Compute token embeddings
        with torch.no_grad():
            embeddings = self.model(**encoded_input.to(self.device)).last_hidden_state[:,0,:]

        # l2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().tolist()

        return embeddings


class OpenAISDKAdapter:
    """
    Adapter for OpenAI API interactions.
    """
    def __init__(self, model_name: str) -> None:
        """
        Initializes the OpenAISDKAdapter with the provided model name.

        Args:
            model_name (str): The OpenAI model name.
        """

        import openai

        self.model_name = model_name
        self.client = openai.OpenAI()

        print(f"OpenAI model '{model_name}' loaded.")

    def get_embedding(self, texts: List[str]) -> Union[List[float], List[List[float]]]:
        """
        Retrieves embeddings for a list of texts using OpenAI API.
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            encoding_format="float",
        )
        outputs = [e.embedding for e in response.data]
        return outputs
