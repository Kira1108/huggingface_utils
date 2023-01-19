from typing import Dict, List
from .base import MetricCompute
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def flatten(list_of_lists:List[List]):
    """Simple flatten function"""
    return [val for sublist in list_of_lists for val in sublist]

class FlatSeqMetric(MetricCompute):


    def call(self, logits, labels) -> Dict[str, float] :
        # compute labels from logits
        preds = np.argmax(logits, axis = -1)

        # mask with -100
        fill_with_null = np.where(np.array(labels) == -100, -100, preds)

        # filter out -100
        pred_labels = [[l for l in label if l!= -100] for label in fill_with_null]
        labels = [[l for l in label if l!= -100] for label in labels]
        pred_labels = flatten(pred_labels)
        labels = flatten(labels)

        preds = np.array(preds)
        return {
            "f1":f1_score(labels, pred_labels, average = 'macro'),
            "accuracy":accuracy_score(labels, pred_labels)
            }
        
if __name__ == "__main__":
    import numpy as np
    logits = np.random.random((10,3))
    labels = np.random.choice([0,1,2], size = (1,10))
    print(np.argmax(logits, axis = -1))
    print(labels[0])
    compute_metrics = FlatSeqMetric()
    compute_metrics((logits, labels))
