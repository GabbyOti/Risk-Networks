import numpy as np
import sklearn.metrics as skm
from collections import defaultdict
import warnings

def confusion_matrix(data,
                     ensemble_states,
                     user_nodes,
                     statuses = ['S', 'E' ,'I' ,'H' ,'R' ,'D'],
                     threshold = 0.5,
                     method='or'):

    """
    Wrapper of `sklearn.metrics.confusion_matrix`.
    Args:
    -----
        ensemble_states: (ensemble_size, 6 * population) `np.array` of the current state of the ensemble ODE system.
        statuses       : `list` of statuses of interest.
        threshold      : float in [0,1] used to determine a binary classification
        method: string, 'sum': means you assign true if the sum exceeds the threshold
                        'or' : means you assign true if either exceeds the threshold
    """
    if user_nodes is not None:
        data = {node: data[node] for node in user_nodes}

    status_catalog = dict(zip(['S', 'E', 'I', 'H', 'R', 'D'], np.arange(6)))
    status_of_interest = np.array([status_catalog[status] for status in statuses])
    if ensemble_states.ndim == 1:
        #in the case of "1" ensemble member - ensure array is 2 dimensional
        ensemble_states = np.array([ensemble_states])
    
    population = len(data)
    ensemble_size = ensemble_states.shape[0]
    user_population = int(ensemble_states.shape[1] / 6)

    ensemble_probabilities = np.zeros((6, user_population))
    
    #obtain the prediction of the ensemble by averaging
    ensemble_probabilities = ensemble_states.reshape(ensemble_size, 6, user_population).mean(axis = 0)
   
    #obtain a binary classification of the prediction
    if method == 'sum':
         #if the sum of the statuses of interest > threshold then we assign true
         classification = ensemble_probabilities[status_of_interest].sum(axis=0)
         classification = (classification > threshold)

    elif method == 'or':
        #if either of the statuses of interest > threshold then we assign true
        classification = (ensemble_probabilities[status_of_interest] > threshold).any(axis=0)
    else:
        raise ValueError("please choose methods from 'sum' (default) or 'or' ")

    #interface for sklearn
    data_statuses      = [8 if status_catalog[status] in status_of_interest else 7 for status in list(data.values())]
    ensemble_statuses  = [8 if positive else 7 for positive in classification]
    labels = [7,8]

    return skm.confusion_matrix(data_statuses, ensemble_statuses, labels = labels)


class PredictedNegativeFraction:
    """
    Container for the Predicted Negative Fraction, based on overall class assignment
    Predicted Negative Fraction = True Negatives + False Negatives/ Total
    """

    def __init__(self, name = 'PredictedNegativeFraction'):
        self.name = name

    def __call__(self,
                 data,
                 ensemble_states,
                 user_nodes,
                 statuses = ['S', 'E', 'I', 'H', 'R', 'D'],
                 threshold = 0.5,
                 method = 'or'):
        """
        Args:
        -----
                data           : dictionary with {node : status}
                ensemble_state : (ensemble size, 5 * population) `np.array` with probabilities
                statuses       : statuses of interest.
        """
        cm = confusion_matrix(data, ensemble_states, user_nodes, statuses,threshold, method)
        tn, fp, fn, tp = cm.ravel()

        return (tn + fn) / (tn + fn + tp + fp)

class PredictedPositiveFraction:
    """
    Container for the Predicted Postive Fraction, based on overall class assignment
    Predicted Postive Fraction  = (True Positives + False Positives) / Total
    """

    def __init__(self, name = 'PredictedPositiveFraction'):
        self.name = name

    def __call__(self,
                 data,
                 ensemble_states,
                 user_nodes,
                 statuses = ['S', 'E', 'I', 'H', 'R', 'D'],
                 threshold = 0.5,
                 method = 'or'):
        """
        Args:
        -----
                data           : dictionary with {node : status}
                ensemble_state : (ensemble size, 5 * population) `np.array` with probabilities
                statuses       : statuses of interest.
        """
        cm = confusion_matrix(data, ensemble_states, user_nodes, statuses, threshold, method)
        tn, fp, fn, tp = cm.ravel()

        return (tp + fp) / (tn + fn + tp + fp)

class Accuracy:
    """
    Container for model accuracy metric. Metric based on overall class assignment.
                Accuracy = (True Positives + True Negatives) / Total Cases
    """

    def __init__(self, name = 'Accuracy'):
        self.name = name

    def __call__(self,
                 data,
                 ensemble_states,
                 user_nodes,
                 statuses = ['S', 'E', 'I', 'H', 'R', 'D'],
                 threshold = 0.5,
                 method = 'or'):
        """
            Args:
            -----
                data           : dictionary with {node : status}
                ensemble_state : (ensemble size, 5 * population) `np.array` with probabilities
                statuses       : statuses of interest.
        """
        cm = confusion_matrix(data, ensemble_states, user_nodes, statuses, threshold, method)
        tn, fp, fn, tp = cm.ravel()
        return (tn+tp) / (tn+fp+fn+tp)
    
class TrueNegativeRate:
    """
    True Negative Rate is the specificity, is the selectivity 
    Container for the TNR metric, based on overall class assignment
    Specificity (True Negative Rate) = True Negatives / (True Negatives + False Positives)
    """

    def __init__(self, name = 'TrueNegativeRate'):
        self.name = name

    def __call__(self,
                 data,
                 ensemble_states,
                 user_nodes,
                 statuses = ['S', 'E', 'I', 'H', 'R', 'D'],
                 threshold = 0.5,
                 method = 'or'):
        """
        Args:
        -----
                data           : dictionary with {node : status}
                ensemble_state : (ensemble size, 5 * population) `np.array` with probabilities
                statuses       : statuses of interest.
        """
        cm = confusion_matrix(data, ensemble_states, user_nodes, statuses, threshold, method)
        tn, fp, fn, tp = cm.ravel()

        #the setting where we cannot measure a negative rate as there are no negative values
        if (tn + fp) == 0: #tn,fp are `int`
            warnings.warn("TrueNegativeRate is returning 0, but is not valid when there are no negative values")
            return 0

        return tn / (tn + fp)

class TruePositiveRate:
    """
    True Positive Rate is the sensitivity, is the recall, is the hit rate.
    Container for model TPR Metric, based on overall class assignment.
    
    Sensitivity (True Positive Rate) = True Positives / (True Positives + False Negatives)
    """

    def __init__(self, name = 'TruePositiveRate'):
        self.name = name

    def __call__(self,
                 data,
                 ensemble_states,
                 user_nodes,
                 statuses = ['S', 'E', 'I', 'H', 'R', 'D'],
                 threshold = 0.5,
                 method = 'or'):
        """
        Args:
        -----
                data           : dictionary with {node : status}
                ensemble_state : (ensemble size, 5 * population) `np.array` with probabilities
                statuses       : statuses of interest.
        """
        cm = confusion_matrix(data, ensemble_states, user_nodes, statuses, threshold, method)
        tn, fp, fn, tp = cm.ravel()

        #the setting where we cannot measure a positive rate as there are no positive values
        if (tp + fn) == 0: #tp, fn are `int`
            warnings.warn("TruePositiveRate is returning 0, but is not valid when there are no positive values")
            return 0
        
        return tp / (tp + fn)
    

class F1Score:
    """
    Container for the F1 score metric. Score used for highly unbalanced data sets.
    Harmonic mean of precision and recall.
            F1 = 2 / ( recall^-1 + precision^-1).
    """

    def __init__(self, name = 'F1 Score'):
        self.name = name

    def __call__(self,
                 data,
                 ensemble_states,
                 user_nodes,
                 statuses = ['E', 'I'],
                 threshold = 0.5,
                 method = 'or'):
        """
        Glossary:
            tn : true negative
            fp : false positive
            fn : false negative
            tp : true positive
        """
        cm = confusion_matrix(data, ensemble_states, user_nodes, statuses, threshold, method)
        tn, fp, fn, tp = cm.ravel()

        #the setting where everything is negative, and captured perfectly 
        if (tp + fp + fn) == 0: #tp, fn are `int`
            warnings.warn("F1Score is returning 0, but is not valid in the current scenario")
            return 0
        
        return 2 * tp / (2 * tp + fp + fn)

class PerformanceTracker:
    """
    Container to track how a classification model behaves over time.
    """
    def __init__(self,
                 metrics   = [TrueNegativeRate(),TruePositiveRate()],
                 user_nodes = None,
                 statuses  = ['E', 'I'],
                 threshold = 0.5,
                 method = 'or' ):
        """
        Args:
        ------
            metrics: list of metrics that can be fed to the wrapper.
            statuses: statuses of interest.
            threshold: a threshold probabilitiy for classification
            method: 'sum' or 'or' to determine how statuses exceed a threshold
        """
        self.statuses  = statuses
        self.user_nodes = user_nodes
        self.metrics   = metrics
        self.threshold = threshold
        self.method = method
        self.performance_track = None
        self.prevalence_track  = None

    def __str__(self):
        """
            Prints current metrics.
        """
        print(" ")
        print("=="*30)
        for kk, metric in enumerate(self.metrics):
            print("[ %s ]                          : %.4f,"%(metric.name, self.performance_track[-1,kk]))
        print("=="*30)
        return ""

    def eval_metrics(self,
                     data,
                     ensemble_states):
        """
        Evaluates each metric in list of metrics.
        Args:
        -----
            data: dictionary with {node : status}
            ensemble_state: (ensemble size, 5 * population) `np.array` with probabilities
        """

        results = [metric(data, ensemble_states, self.user_nodes, self.statuses, self.threshold, self.method) for metric in self.metrics]
        if self.performance_track is None:
            self.performance_track = np.array(results).reshape(1, len(self.metrics))
        else:
            self.performance_track = np.vstack([self.performance_track, results])

    def eval_prevalence(self, data):
        """
        Evaluates the prevalence of the status of interest in `self.statuses`.
        If multiple, it combines them as a single status.
            Args:
            -----
                data: dictionary with {node : status}
        """
        
        
        status_catalog = dict(zip(['S','E', 'I', 'H', 'R', 'D'], np.arange(6)))
        population = len(data)
     
        a, b = np.unique([v for v in data.values()], return_counts=True)
        status_counts = defaultdict(int, zip(a, b))

        prevalence = np.array([status_counts[status] for status in self.statuses]).sum()/population

        if self.prevalence_track is None:
            self.prevalence_track = np.array(prevalence)
        else:
            self.prevalence_track = np.hstack([self.prevalence_track, prevalence])

    def update(self, data, ensemble_states):
        """
        Evaluates both the prevalence of the status of interest in `self.statuses`,
        and the performance metrics given a snapshot of the current state of the
        sytem (`kinetic model`) and the model (ensemble 'master equations').

            Args:
            -----
                data: dictionary with {node : status}
                ensemble_state: (ensemble size, 6 * population) `np.array` with probabilities
        """
        self.eval_metrics(data, ensemble_states)
        self.eval_prevalence(data)


