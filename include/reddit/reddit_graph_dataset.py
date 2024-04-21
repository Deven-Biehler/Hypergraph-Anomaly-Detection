
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
            "num_vertices": 7025,
            "num_edges": 2165,
            "dim_features": 7138,
            "features": {
                "upon": [
                    {
                        "filename": "features.pkl", 
                        "md5": "c4e93fd0190a09eec1aa6791572892a0",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [
                    {
                        "filename": "edgelist.pkl",
                        "md5": "3e35ec8caacb1b2986e5a443f5ffdf1d",
                    }
                ],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [
                    {
                        "filename": "labels.pkl",
                        "md5": "5d8ab83c1ac4e2d92ea2f749de5f5822",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [
                    {
                        "filename": "train_mask.pkl",
                        "md5": "54e4610a1f41aee91bd9c7bf4945f3c7",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [
                    {
                        "filename": "val_mask.pkl",
                        "md5": "20d1ea8c050610e8c5727e18654601f9",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [
                    {
                        "filename": "test_mask.pkl",
                        "md5": "9ddda70608d879355925c36bc1003f0f",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }