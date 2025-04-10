# traffick_fluo/model/registry.py

from sklearn.ensemble import RandomForestClassifier
# Optional future imports:
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier

from typing import Type, Dict
from sklearn.base import ClassifierMixin

MODEL_REGISTRY: Dict[str, Type[ClassifierMixin]] = {
    "RandomForestClassifier": RandomForestClassifier,
    # "LogisticRegression": LogisticRegression,
    # "XGBClassifier": XGBClassifier,
}
