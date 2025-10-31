# Summary of file: enVector SDK Adapter(enVector APIs Caller)

from typing import Union, List, Dict, Any
import numpy as np
import es2  # pip install es2
from es2.crypto.block import CipherBlock

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
KEY_PATH = SCRIPT_DIR / "keys"
KEY_PATH.mkdir(exist_ok=True)

class EnVectorSDKAdapter:
    """
    Adapter class to interact with the enVector SDK.
    """
    def __init__(
            self,
            endpoint: str,
            port: int,
            key_id: str,
            eval_mode: str
        ):
        """
        Initializes the EnVectorSDKAdapter with an optional endpoint.

        Args:
            endpoint (Optional[str]): The endpoint URL for the enVector SDK.
            port (Optional[int]): The port number for the enVector SDK.
        """
        self.endpoint = endpoint
        self.port = port
        es2.init(host=self.endpoint, port=self.port, key_path=str(KEY_PATH), key_id=key_id, eval_mode=eval_mode, auto_key_setup=True)

    #------------------- Insert ------------------#

    def call_insert(self, index_name: str, vectors: Union[List[List[float]], List[CipherBlock]], metadata: List[Any] = None):
        """
        Calls the enVector SDK to perform an insert operation.

        Args:
            vectors (Union[List[List[float]], List[CipherBlock]]): The list of vectors to insert.
            metadata (List[Any], optional): The list of metadata associated with the vectors. Defaults to None.

        Returns:
            Dict[str, Any]: If succeed, converted format of the insert results. Otherwise, error message.
        """
        try:
            results = self.invoke_insert(index_name=index_name, vectors=vectors, metadata=metadata)
            return self._to_json_available({"ok": True, "results": results})
        except Exception as e:
            # Handle exceptions and return an appropriate error message
            return {"ok": False, "error": repr(e)}

    def invoke_insert(self, index_name: str, vectors: Union[List[List[float]], List[CipherBlock]], metadata: List[Any] = None):
        """
        Invokes the enVector SDK's insert functionality.

        Args:
            index_name (str): The name of the index to insert into.
            vectors (Union[List[List[float]], List[CipherBlock]]): The list of vectors to insert.
            metadata (List[Any], optional): The list of metadata associated with the vectors. Defaults to None.

        Returns:
            Any: Raw insert results from the enVector SDK.
        """
        index = es2.Index(index_name)  # Create an index instance with the given index name
        # Insert vectors with optional metadata
        return index.insert(data=vectors, metadata=metadata) # Return list of inserted vectors' IDs

    #------------------- Search ------------------#

    def call_search(self, index_name: str, query: Union[List[float], List[List[float]]], topk: int) -> Dict[str, Any]:
        """
        Calls the enVector SDK to perform a search operation.

        Args:
            index_name (str): The name of the index to search.
            query (Union[List[float], List[List[float]]]): The search query.
            topk (int): The number of top results to return.

        Returns:
            Dict[str, Any]: If succeed, converted format of the search results. Otherwise, error message.
        """
        try:
            results = self.invoke_search(index_name=index_name, query=query, topk=topk)
            return self._to_json_available({"ok": True, "results": results})
        except Exception as e:
            # Handle exceptions and return an appropriate error message
            return {"ok": False, "error": repr(e)}

    def invoke_search(self, index_name: str, query: Union[List[float], List[List[float]]], topk: int):
        """
        Invokes the enVector SDK's search functionality.

        Args:
            index_name (str): The name of the index to search.
            query (Union[List[float], List[List[float]]]): The search query.
            topk (int): The number of top results to return.

        Returns:
            Any: Raw search results from the enVector SDK.
        """
        index = es2.Index(index_name)  # Create an index instance with the given index name
        # Search with the provided query and topk. Fixed output_fields parameter for now.
        return index.search(query, top_k=topk, output_fields=["metadata"])

    @staticmethod
    def _to_json_available(obj: Any) -> Any:
        """
        Converts an object to a JSON-serializable format if possible.

        Args:
            obj (Any): The object to convert.

        Returns:
            Any: The JSON-serializable representation of the object, or the original object if conversion is not possible.
        """
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): EnVectorSDKAdapter._to_json_available(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [EnVectorSDKAdapter._to_json_available(item) for item in obj]
        for attr in ("model_dump", "dict", "to_dict"):
            if hasattr(obj, attr):
                try:
                    return EnVectorSDKAdapter._to_json_available(getattr(obj, attr)())
                except Exception:
                    pass
        if hasattr(obj, "__dict__"):
            try:
                return {k: EnVectorSDKAdapter._to_json_available(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
            except Exception:
                pass
        return repr(obj)
