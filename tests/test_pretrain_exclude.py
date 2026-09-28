"""
CPU-only unit tests for collect_pretrain_files()'s manifest "exclude" field
(row-level filtering of pretraining files).

No pytest in this env -- plain asserts, run directly:
    python tests/test_pretrain_exclude.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.dataloader import collect_pretrain_files


def _make_tree(root):
    """ds/USER00001/{wrist,thigh}, ds/USER00002/wrist -- ACC/X.npy in each."""
    for user, sensors in [("USER00001", ["wrist", "thigh"]), ("USER00002", ["wrist"])]:
        for sensor in sensors:
            d = os.path.join(root, "ds", user, sensor, "ACC")
            os.makedirs(d)
            np.save(os.path.join(d, "X.npy"), np.zeros((2, 3, 150), dtype=np.float32))


def _rows(paths, root):
    """X.npy paths -> sorted "USERxxxxx/<sensor>" strings."""
    return sorted(os.path.relpath(p, os.path.join(root, "ds")).rsplit("/ACC/", 1)[0] for p in paths)


def test_no_exclude_unchanged():
    with tempfile.TemporaryDirectory() as root:
        _make_tree(root)
        by_name, _ = collect_pretrain_files(["ds"], data_root=root)
        by_dict, _ = collect_pretrain_files([{"dataset": "ds", "data_root": root}])
        assert sorted(by_name) == sorted(by_dict)
        assert _rows(by_name, root) == ["USER00001/thigh", "USER00001/wrist", "USER00002/wrist"]


def test_exclude_user_sensor():
    with tempfile.TemporaryDirectory() as root:
        _make_tree(root)
        paths, _ = collect_pretrain_files(
            [{"dataset": "ds", "data_root": root, "exclude": ["USER00001/wrist"]}])
        assert _rows(paths, root) == ["USER00001/thigh", "USER00002/wrist"]


def test_exclude_whole_user():
    with tempfile.TemporaryDirectory() as root:
        _make_tree(root)
        paths, _ = collect_pretrain_files(
            [{"dataset": "ds", "data_root": root, "exclude": ["USER00001"]}])
        assert _rows(paths, root) == ["USER00002/wrist"]


def test_exclude_unmatched_raises():
    with tempfile.TemporaryDirectory() as root:
        _make_tree(root)
        for bad in ["USER00009", "USER00001/ankle"]:
            try:
                collect_pretrain_files([{"dataset": "ds", "data_root": root, "exclude": [bad]}])
            except ValueError as e:
                assert bad in str(e)
            else:
                raise AssertionError(f"exclude {bad!r} should raise")


def test_missing_dataset_still_skipped():
    with tempfile.TemporaryDirectory() as root:
        _make_tree(root)
        paths, skipped = collect_pretrain_files(
            [{"dataset": "nope", "data_root": root, "exclude": ["USER00001"]}])
        assert paths == [] and skipped == ["nope"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
