from typing import List, Tuple


def get_confusion_matrix(
        y_true: List[int], y_pred: List[int], num_classes: int,
) -> List[List[int]]:
    """
    Generate a confusion matrix in a form of a list of lists. 

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values
    :param num_classes: number of supported classes

    :return: confusion matrix
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Invalid input shapes!")

    if len(y_true) != num_classes:
        raise ValueError("Invalid prediction classes!")

    cm = [[0] * num_classes for _ in range(num_classes)]

    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    return cm


def get_quality_factors(
        y_true: List[int],
        y_pred: List[int],
) -> Tuple[int, int, int, int]:
    """
    Calculate True Negative, False Positive, False Negative and True Positive 
    metrics basing on the ground truth and predicted lists.

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: a tuple of TN, FP, FN, TP
    """

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for truth, pred in zip(y_true, y_pred):
        if truth == 0:
            if pred == 0:
                tn += 1
            else:
                fp += 1
        else:
            if pred == 0:
                fn += 1
            else:
                tp += 1

    return tn, fp, fn, tp



def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the accuracy for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: accuracy score
    """

    tn, fp, fn, tp = get_quality_factors(y_true, y_pred)

    return (tn + tp) / (tn + tp + fp + fn)


def precision_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the precision for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: precision score
    """
    tn, fp, fn, tp = get_quality_factors(y_true, y_pred)

    return tp / (tp + fp)


def recall_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the recall for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: recall score
    """
    tn, fp, fn, tp = get_quality_factors(y_true, y_pred)

    return tp / (tp + fn)


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the F1-score for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: F1-score
    """

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return 2 * precision * recall / (precision + recall)
