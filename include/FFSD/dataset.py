
from typing import Optional
import pickle
from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor
from functools import partial
from dhg.data import BaseData

class FFSD(BaseData):
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("ffsd", data_root)
        self._content = {
            "num_classes": 2,
            "num_vertices": 29643,
            "num_edges": 626,
            "dim_features": 6,
            "features": {
                "upon": [
                    {
                        "filename": "features.pkl", 
                        "md5": "9e9e536593973c33626aecff81901133",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [
                    {
                        "filename": "edgelist.pkl",
                        "md5": "7338ba44b09e7c40dec8537e246cc1c0",
                    }
                ],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [
                    {
                        "filename": "labels.pkl",
                        "md5": "eb9585210017f02c1c4016ad695ea84b",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [
                    {
                        "filename": "train_mask.pkl",
                        "md5": "fe191269320e40647c5072869107dd6c",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [
                    {
                        "filename": "val_mask.pkl",
                        "md5": "1b4530629f6f7843dad749bb4f52d104",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [
                    {
                        "filename": "test_mask.pkl",
                        "md5": "8402d7ade1753a8caa3b9f41076a8282",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }