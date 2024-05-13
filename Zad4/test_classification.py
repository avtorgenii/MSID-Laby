"""Homework for lab_4."""
from classification import (
    get_confusion_matrix,
    get_quality_factors,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import numpy as np
import pytest


@pytest.mark.parametrize(
    "y_true, y_pred, num_classes, proper_cm",
    [
        (
                [0, 1, 2, 0, 1, 2, 1, 1, 2],
                [2, 0, 2, 1, 0, 2, 2, 0, 2, ],
                3,
                [
                    [0, 1, 1],
                    [3, 0, 1],
                    [0, 0, 3],
                ]
        ),

        (
                [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, ],
                [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, ],
                2,
                [
                    [5, 3],
                    [1, 3],
                ],
        ),
        (
                [0, 2, 0, 0, 1, 4, 0, ],
                [2, 3, 1, 0, 4, 0, 2, ],
                5,
                [
                    [1, 1, 2, 0, 0, ],
                    [0, 0, 0, 0, 1, ],
                    [0, 0, 0, 1, 0, ],
                    [0, 0, 0, 0, 0, ],
                    [1, 0, 0, 0, 0, ],
                ]
        )
    ],
)
def test_confusion_matrix(y_true, y_pred, num_classes, proper_cm):
    assert proper_cm == get_confusion_matrix(y_true, y_pred, num_classes)


@pytest.mark.parametrize(
    "y_true, y_pred, num_classes",
    [
        ([0, 0, 0], [0, 0, 0, 0], 13),
        ([0, 0, 0, 0], [0, 0, 0], 13),
    ],
)
def test_cm_incorrect_shapes(y_true, y_pred, num_classes):
    with pytest.raises(ValueError) as excinfo:
        get_confusion_matrix(y_true, y_pred, num_classes)
    assert "Invalid input shapes!" in str(excinfo.value)


@pytest.mark.parametrize(
    "y_true, y_pred, num_classes",
    [
        ([0, 1, 2, 2], [0, 5, 1, 2], 3),
        ([0, 1, 2, 2], [0, 61, 1, 2], 3),
    ],
)
def test_cm_incorrect_prediction_classes(y_true, y_pred, num_classes):
    with pytest.raises(ValueError) as excinfo:
        get_confusion_matrix(y_true, y_pred, num_classes)
    assert "Invalid prediction classes!" in str(excinfo.value)


@pytest.mark.parametrize(
    "y_true, y_pred, quality_factors",
    [
        (
                [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, ],
                [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, ],
                (3, 1, 1, 5,),
        ),
        (
                [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, ],
                [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, ],
                (3, 5, 1, 3,),
        ),
        (
                [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, ],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, ],
                (4, 3, 2, 1,),
        ),
    ]
)
def test_get_quality_factors(y_true, y_pred, quality_factors):
    assert quality_factors == get_quality_factors(y_pred, y_true)


@pytest.mark.parametrize(
    "y_true, y_pred, accuracy",
    [
        (
                [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, ],
                [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, ],
                0.8,
        ),
        (
                [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, ],
                [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, ],
                0.5
        ),
        (
                [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, ],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, ],
                0.5
        ),
    ],
)
def test_accuracy_score(y_true, y_pred, accuracy):
    np.testing.assert_almost_equal(
        desired=accuracy,
        actual=accuracy_score(y_true, y_pred),
        decimal=5
    )


@pytest.mark.parametrize(
    "y_true, y_pred, precision",
    [
        (
                [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, ],
                [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, ],
                0.8333333333
        ),
        (
                [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, ],
                [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, ],
                0.75,
        ),
        (
                [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, ],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, ],
                0.33333333333
        ),
    ],
)
def test_precision_score(y_true, y_pred, precision):
    np.testing.assert_almost_equal(
        desired=precision,
        actual=precision_score(y_true, y_pred),
        decimal=5
    )


@pytest.mark.parametrize(
    "y_true, y_pred, recall",
    [
        (
                [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, ],
                [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, ],
                0.8333333333,
        ),
        (
                [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, ],
                [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, ],
                0.375,
        ),
        (
                [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, ],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, ],
                0.25,
        ),
    ],
)
def test_recall_score(y_true, y_pred, recall):
    np.testing.assert_almost_equal(
        desired=recall,
        actual=recall_score(y_true, y_pred),
        decimal=5,
    )


@pytest.mark.parametrize(
    "y_true, y_pred, f1",
    [
        (
                [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, ],
                [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, ],
                0.8333333333,
        ),
        (
                [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, ],
                [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, ],
                0.5,
        ),
        (
                [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, ],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, ],
                0.2857142857,
        ),
    ],
)
def test_get_f1_score(y_true, y_pred, f1):
    np.testing.assert_almost_equal(
        desired=f1,
        actual=f1_score(y_true, y_pred),
        decimal=5,
    )
