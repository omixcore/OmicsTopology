"""Reading matrices and group files, writing result tables."""

import os

import numpy as np
import pytest

from omictopo.io import read_groups, read_matrix, write_matrix, write_table


def test_matrix_round_trip(tmp_path):
    path = str(tmp_path / "expr.tsv")
    X = np.array([[1.0, 2.5, 3.0], [4.0, 5.5, 6.0]])
    write_matrix(path, X, rownames=["s1", "s2"], colnames=["g1", "g2", "g3"])

    back, rows, cols = read_matrix(path)
    assert rows == ["s1", "s2"]
    assert cols == ["g1", "g2", "g3"]
    assert np.allclose(back, X)


def test_csv_sniffing(tmp_path):
    path = str(tmp_path / "expr.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",g1,g2\ns1,1,2\ns2,3,4\n")
    X, rows, cols = read_matrix(path)
    assert X.shape == (2, 2)
    assert cols == ["g1", "g2"]
    assert np.allclose(X, [[1, 2], [3, 4]])


def test_missing_values_become_nan(tmp_path):
    path = str(tmp_path / "m.tsv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("f1\tf2\ns1\t1\tNA\ns2\t2\t3\n")
    X, _, _ = read_matrix(path)
    assert np.isnan(X[0, 1])
    assert X[1, 1] == 3.0


def test_groups_single_column(tmp_path):
    path = str(tmp_path / "g.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("control\ncontrol\nsalt\n")
    ids, labels = read_groups(path)
    assert ids == []
    assert labels == ["control", "control", "salt"]


def test_groups_two_columns_with_header(tmp_path):
    path = str(tmp_path / "g.tsv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("sample\tgroup\ns1\tcontrol\ns2\tsalt\n")
    ids, labels = read_groups(path)
    assert ids == ["s1", "s2"]
    assert labels == ["control", "salt"]


def test_write_table(tmp_path):
    path = str(tmp_path / "out" / "summary.tsv")
    write_table(path, [{"state": "a", "value": 1.5}, {"state": "b", "value": None}])
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert lines[0].split("\t")[:2] == ["state", "value"]
    assert lines[1].split("\t") == ["a", "1.5"]
    assert lines[2].split("\t") == ["b", ""]


def test_write_table_handles_infinite_values(tmp_path):
    path = str(tmp_path / "inf.tsv")
    write_table(path, [{"death": float("inf"), "birth": 0.0}])
    with open(path, encoding="utf-8") as fh:
        assert "inf" in fh.read()


def test_read_matrix_rejects_empty_file(tmp_path):
    path = str(tmp_path / "empty.tsv")
    open(path, "w").close()
    with pytest.raises(ValueError, match="empty"):
        read_matrix(path)
