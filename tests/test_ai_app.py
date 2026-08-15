from __future__ import annotations

from pathlib import Path

import pytest

from fairvote.study import METHOD_LABELS

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_runs_without_recommender_data_and_explains_how_to_generate_it(
    tmp_path,
    monkeypatch,
) -> None:
    streamlit = pytest.importorskip("streamlit")
    monkeypatch.chdir(tmp_path)
    streamlit.cache_resource.clear()

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    assert not app.exception
    assert "AI-assisted estimator recommendation" in [element.value for element in app.subheader]
    assert "python -m experiments.train_selector" in [element.value for element in app.code]

    rendered = "\n".join(str(frame.value) for frame in app.dataframe)
    for label in METHOD_LABELS.values():
        assert label in rendered


def test_app_runs_the_recommender_when_selector_data_exists(
    tmp_path,
    monkeypatch,
) -> None:
    streamlit = pytest.importorskip("streamlit")
    monkeypatch.chdir(tmp_path)
    streamlit.cache_resource.clear()

    from streamlit.testing.v1 import AppTest

    from fairvote.ai.features import build_selector_dataset

    dataset_path = tmp_path / "results" / "ai" / "selector_dataset.csv"
    dataset_path.parent.mkdir(parents=True)
    build_selector_dataset(quick=True).to_csv(dataset_path, index=False)

    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    assert not app.exception

    rendered = "\n".join(str(element.value) for element in app.markdown)
    assert "Recommended estimator:" in rendered or "Approximate tie between" in rendered

    assert "python -m experiments.train_selector" not in [str(element.value) for element in app.code]
