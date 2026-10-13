"""Command-line entry point."""

import os

import numpy as np

from omictopo.cli import main
from omictopo.datasets import two_state_omics
from omictopo.io import write_matrix


def test_version_command(capsys):
    assert main(["version"]) == 0
    assert "omictopo" in capsys.readouterr().out


def test_full_run_from_files(tmp_path, capsys):
    X, groups, _ = two_state_omics(n_genes=200, n_per_state=20, seed=0)
    matrix = str(tmp_path / "expr.tsv")
    groupfile = str(tmp_path / "groups.tsv")
    outdir = str(tmp_path / "results")

    write_matrix(matrix, X, rownames=[f"s{i}" for i in range(X.shape[0])],
                 colnames=[f"g{j}" for j in range(X.shape[1])])
    with open(groupfile, "w", encoding="utf-8") as fh:
        fh.write("sample\tgroup\n")
        for i, g in enumerate(groups):
            fh.write(f"s{i}\t{'control' if g == 0 else 'salt'}\n")

    code = main([
        "run",
        "--matrix", matrix,
        "--groups", groupfile,
        "--labels", "control,salt_stress",
        "--outdir", outdir,
        "--max-points", "20",
        "--max-simplices", "20000",
        "--n-permutations", "0",
    ])

    out = capsys.readouterr().out
    assert code == 0
    assert "salt_stress" in out
    assert os.path.exists(os.path.join(outdir, "topology_summary.tsv"))

    with open(os.path.join(outdir, "topology_summary.tsv"), encoding="utf-8") as fh:
        header = fh.readline()
    assert "h1_total_persistence" in header


def test_sample_order_is_realigned(tmp_path, capsys):
    """Rows may appear in any order as long as the id columns match."""
    X, groups, _ = two_state_omics(n_genes=120, n_per_state=15, seed=1)
    matrix = str(tmp_path / "expr.tsv")
    groupfile = str(tmp_path / "groups.tsv")
    order = np.arange(X.shape[0])[::-1]  # reversed

    write_matrix(matrix, X, rownames=[f"s{i}" for i in range(X.shape[0])],
                 colnames=[f"g{j}" for j in range(X.shape[1])])
    with open(groupfile, "w", encoding="utf-8") as fh:
        for i in order:
            fh.write(f"s{i}\t{'control' if groups[i] == 0 else 'salt'}\n")

    code = main([
        "run", "--matrix", matrix, "--groups", groupfile,
        "--labels", "control,salt", "--max-points", "15",
        "--max-simplices", "20000", "--no-test",
    ])
    assert code == 0
    assert "control" in capsys.readouterr().out


def test_backend_and_statistic_flags_are_accepted(tmp_path, capsys):
    """--backend / --statistic must reach run_states rather than error out."""
    X, groups, _ = two_state_omics(n_genes=120, n_per_state=15, seed=3)
    matrix = str(tmp_path / "expr.tsv")
    groupfile = str(tmp_path / "groups.tsv")
    write_matrix(matrix, X, rownames=[f"s{i}" for i in range(X.shape[0])],
                 colnames=[f"g{j}" for j in range(X.shape[1])])
    with open(groupfile, "w", encoding="utf-8") as fh:
        fh.write("sample\tgroup\n")
        for i, g in enumerate(groups):
            fh.write(f"s{i}\t{'control' if g == 0 else 'salt'}\n")

    code = main([
        "run", "--matrix", matrix, "--groups", groupfile,
        "--labels", "control,salt", "--max-points", "15",
        "--max-simplices", "20000", "--no-test",
        "--backend", "builtin", "--statistic", "bottleneck",
    ])
    assert code == 0
    assert "control" in capsys.readouterr().out


def test_unknown_sample_is_reported(tmp_path):
    X, groups, _ = two_state_omics(n_genes=60, n_per_state=8, seed=2)
    matrix = str(tmp_path / "expr.tsv")
    groupfile = str(tmp_path / "groups.tsv")
    write_matrix(matrix, X, rownames=[f"s{i}" for i in range(X.shape[0])],
                 colnames=[f"g{j}" for j in range(X.shape[1])])
    with open(groupfile, "w", encoding="utf-8") as fh:
        fh.write("sample\tgroup\nghost\tcontrol\n")
        for i in range(1, X.shape[0]):
            fh.write(f"s{i}\tcontrol\n")

    try:
        main(["run", "--matrix", matrix, "--groups", groupfile])
    except SystemExit as exc:
        assert "unknown samples" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for an unknown sample id")
