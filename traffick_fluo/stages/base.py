# traffick_fluo/stages/base.py

from abc import ABC, abstractmethod

class BaseStage(ABC):
    @abstractmethod
    def extract_features(self, image_path):
        pass

    @abstractmethod
    def score(self, config):
        from traffick_fluo.model.apply import score_features
        score_features(self, config)
