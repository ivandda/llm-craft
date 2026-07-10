import json

from src.data import import_final10k_to_postgres as importer


class FakeConnection:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, statement, params=None):
        self.statements.append(statement)
        return self


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_importer_ignores_rejected_jsonl(tmp_path, monkeypatch, capsys):
    record = {
        "input_a": "Fire",
        "input_b": "Water",
        "candidate_outputs": [{"output": "Steam", "source": "observed", "rank": 1}],
    }
    for split in importer.SPLITS:
        write_jsonl(tmp_path / f"{split}.jsonl", [record])
    write_jsonl(tmp_path / "rejected.jsonl", [{"input_a": "bad", "input_b": "row"}])

    connection = FakeConnection()
    monkeypatch.setattr(importer, "connect", lambda: connection)
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_final10k_to_postgres",
            "--dataset-dir",
            str(tmp_path),
            "--replace-dataset",
            "final-10k",
        ],
    )

    importer.main()

    output = json.loads(capsys.readouterr().out)
    sql = "\n".join(connection.statements)
    assert output == {"dataset_name": "final-10k", "train": 1, "dev": 1, "test": 1, "rejected": 0}
    assert "dataset_rejections" not in sql
    assert "rejected_count = 0" in sql
