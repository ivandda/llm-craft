import sys
import zipfile

from src.sft.create_colab_zip import DEFAULT_STRUCTURED_EVAL_FILE, main


def test_default_structured_eval_file_is_dev_set():
    assert DEFAULT_STRUCTURED_EVAL_FILE == "datasets/processed/eval_dev_all.jsonl"


def test_create_colab_zip_includes_structured_dev_eval_set(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("guide\n", encoding="utf-8")
    (tmp_path / "artifacts" / "data").mkdir(parents=True)
    (tmp_path / "artifacts" / "data" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "artifacts" / "data" / "dev.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "datasets" / "processed").mkdir(parents=True)
    (tmp_path / "datasets" / "processed" / "eval_dev_all.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_colab_zip",
            "--output-path",
            "out/colab.zip",
            "--train-file",
            "artifacts/data/train.jsonl",
            "--eval-file",
            "artifacts/data/dev.jsonl",
            "--eval-set-file",
            "datasets/processed/eval_dev_all.jsonl",
        ],
    )

    main()

    with zipfile.ZipFile(tmp_path / "out" / "colab.zip") as zip_handle:
        assert "llm-craft-colab/datasets/processed/eval_dev_all.jsonl" in zip_handle.namelist()
