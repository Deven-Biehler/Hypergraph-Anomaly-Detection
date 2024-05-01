
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
            "num_vertices": 20046,
            "num_edges": 3016,
            "dim_features": 38758,
            "features": {
                "upon": [
                    {
                        "filename": "features.pkl", 
                        "md5": "f85c671830b36d027d2c35f17e7f19d4",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [
                    {
                        "filename": "edgelist.pkl",
                        "md5": "50cb5eb3053f832d436b58e234c87bc6",
                    }
                ],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [
                    {
                        "filename": "labels.pkl",
                        "md5": "2710b39001f857eb50a089171c8a80e3",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [
                    {
                        "filename": "train_mask.pkl",
                        "md5": "fedd0650f731ac39b2e09d7a13b951f9",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [
                    {
                        "filename": "val_mask.pkl",
                        "md5": "e24dd1c41d9a425144f17b6a64b3acc8",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [
                    {
                        "filename": "test_mask.pkl",
                        "md5": "b1fe02b4bc6a4991bff77aebba28961f",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }