from abc import ABC, abstractmethod
from typing import Dict, Tuple

class MetricCompute(ABC):

    @abstractmethod
    def call(self, logits, labels) -> Dict[str, str] :
        pass

    def __call__(self, logits_and_labels:Tuple) -> Dict[str, str]:
        logits, labels = logits_and_labels
        return self.call(logits, labels)
