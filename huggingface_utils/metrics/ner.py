from .base import MetricCompute
import evaluate
import numpy as np

class NerMetric(MetricCompute):
    
    """
    NerMetric computes metrics of NER task.
    NerMetric object is a callable object that takes logits and labels as input
    You can pass this object to huggingface Trainer constructor
    """

    def __init__(self, label_names:list):
        self.label_names = label_names
        self.metric = evaluate.load('seqeval')

    def call(self, logits, labels):
        
        # compute integer labels for (N,T,K) data.
        pred_labels = np.argmax(logits, axis = -1)

        # fill -100 on unwanted positions
        # filter out -100 labels on both
        # pred_labels and labels array
        fill_with_null = np.where(labels == -100, -100, pred_labels)
        pred_labels = [[self.label_names[l] for l in ele if l!= -100] for ele in fill_with_null]
        labels = [[self.label_names[l] for l in ele if l!= -100] for ele in labels]

        # compute metrics
        result = self.metric.compute(predictions = pred_labels, references = labels)

        # filter out unwanted information
        return {
            "precision":result['overall_precision'],
            "recall":result['overall_recall'],
            "f1":result['overall_f1'],
            "accuracy":result['overall_accuracy']
        }
        
if __name__ == "__main__":
    import numpy as np

    def fake_labels_and_logits(N, T, K):
        labels = np.random.randint(0,K, size = (N,T))
        labels[:,0] = -100
        labels[:,-1] = -100
        labels[-1,-2:] = -100
        labels[-2,-4:] = -100
        logits = np.random.random((N,T,K))
        return labels, logits

    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    labels, logits = fake_labels_and_logits(100, 20, len(label_names))
    
    
    compute_metrics = NerMetric(label_names)
    print(compute_metrics((logits, labels)))
    