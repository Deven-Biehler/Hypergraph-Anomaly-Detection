
from typing import Optional
import pickle
from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor
from functools import partial
from dhg.data import BaseData

class Reddit(BaseData):
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("reddit", data_root)
        self._content = {
            "num_classes": 2,
            "num_vertices": 965,
            "num_edges": 487,
            "dim_features": 983,
            "features": {
                "upon": [
                    {
                        "filename": "features.pkl", 
                        "md5": "363c6e8f3e24453b9345341561be8953",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [
                    {
                        "filename": "edgelist.pkl",
                        "md5": "43163376732fb650adb954191ca61326",
                    }
                ],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [
                    {
                        "filename": "labels.pkl",
                        "md5": "65ba3726423a04bfe23022da5e87b352",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [
                    {
                        "filename": "train_mask.pkl",
                        "md5": "c22c3436492526884815c8aea3857b6a",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [
                    {
                        "filename": "val_mask.pkl",
                        "md5": "16d10cd937e188013850f04bfda059f6",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [
                    {
                        "filename": "test_mask.pkl",
                        "md5": "e438c1b0df980735cb0ee242328f1e89",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }