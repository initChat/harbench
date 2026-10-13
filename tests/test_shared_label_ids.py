"""Shared-class scoring: _shared_label_ids + train_model's val/test label overrides.
    python tests/test_shared_label_ids.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from finetune import _shared_label_ids


def test_int_users_intersection():
    Y = np.array([0, 1, 2, 0, 1, 3])
    U = np.array([1, 1, 1, 2, 2, 2])
    assert _shared_label_ids(Y, U, [1, 2]) == [0, 1]  # 2 only in user 1, 3 only in user 2


def test_str_users_and_restrict():
    Y = np.array([0, 1, 2, 5, 0, 1, 2, 5])
    U = np.array(["t::1"] * 4 + ["t::2"] * 4)
    assert _shared_label_ids(Y, U, ["t::1", "t::2"]) == [0, 1, 2, 5]
    assert _shared_label_ids(Y, U, ["t::1", "t::2"], restrict_ids=[1, 2, 7]) == [1, 2]


def test_missing_users_returns_none():
    assert _shared_label_ids(np.array([0]), np.array([1]), [9]) is None
    assert _shared_label_ids(np.array([0]), np.array([1]), None) is None


if __name__ == "__main__":
    test_int_users_intersection(); test_str_users_and_restrict(); test_missing_users_returns_none()
    print("all shared_label_ids tests passed")
