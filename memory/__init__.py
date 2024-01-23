from typing import Callable, Any, Dict, List, Optional

from pydantic import BaseModel, Extra, Field
from langchain.retrievers import SVMRetriever, KNNRetriever
from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings, HuggingFaceEmbeddings
from pydantic import BaseModel
from langchain.embeddings.base import Embeddings

from .episode import Trajectory

# COPIED FROM HUGGINGFACE EMDDEDING CLASS
class GPT4ALLEmbeddings(BaseModel, Embeddings):
    """Wrapper around sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any  #: :meta private:
    model_name: str = 'gpt4all'
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        try:
            from gpt4all import Embed4All

        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence_transformers`."
            ) from exc

        self.client = Embed4All()

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = [self.client.embed(text) for text in texts]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.client.embed(text)
        return embedding


def choose_embedder(key: str) -> Callable:
    if key == 'openai':
        return OpenAIEmbeddings
    if key == 'llama':
        return LlamaCppEmbeddings
    if key == 'gpt4all':
        return GPT4ALLEmbeddings
    return HuggingFaceEmbeddings

def choose_retriever(key: str) -> Callable:
    if key == 'knn':
        return KNNRetriever
    if key == 'svm':
        return SVMRetriever
    return KNNRetriever

EMBEDDERS = choose_embedder
RETRIEVERS = choose_retriever