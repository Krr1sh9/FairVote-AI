from __future__ import annotations

from pathlib import Path

import pytest

from fairvote.study import METHOD_LABELS

# The Streamlit application is loaded from the repository root rather than from the temporary working directories used by the tests.
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_runs_without_recommender_data_and_explains_how_to_generate_it(
    tmp_path,
    monkeypatch,
) -> None:
    # The test is skipped when Streamlit is unavailable rather than failing because of an optional application dependency.
    streamlit = pytest.importorskip("streamlit")

    # Running from an empty temporary directory means the app's relative selector-dataset path does not exist for this test.
    monkeypatch.chdir(tmp_path)

    # Clearing Streamlit's resource cache prevents a selector fitted in another test run from being reused here.
    streamlit.cache_resource.clear()

    from streamlit.testing.v1 import AppTest

    # Streamlit's testing interface executes the application without starting an external web server.
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    # The statistical application must complete without an uncaught Streamlit exception even though the recommender dataset is absent.
    assert not app.exception

    # The page must still show the recommender section and the command that can be used to generate its required dataset.
    assert "AI-assisted estimator recommendation" in [element.value for element in app.subheader]
    assert "python -m experiments.train_selector" in [element.value for element in app.code]

    # The rendered dataframes must still contain the labels of all three statistical estimators.
    rendered = "\n".join(str(frame.value) for frame in app.dataframe)
    for label in METHOD_LABELS.values():
        assert label in rendered


def test_app_runs_the_recommender_when_selector_data_exists(
    tmp_path,
    monkeypatch,
) -> None:
    # Streamlit is again treated as an optional test dependency.
    streamlit = pytest.importorskip("streamlit")

    # The temporary directory becomes the application's working directory so a test-specific results tree can be supplied.
    monkeypatch.chdir(tmp_path)

    # Clearing cached resources ensures the app loads and fits from the selector dataset created specifically for this test.
    streamlit.cache_resource.clear()

    from streamlit.testing.v1 import AppTest

    from fairvote.ai.features import build_selector_dataset

    # A deterministic quick selector dataset is written to the relative path expected by the application.
    dataset_path = tmp_path / "results" / "ai" / "selector_dataset.csv"
    dataset_path.parent.mkdir(parents=True)
    build_selector_dataset(quick=True).to_csv(dataset_path, index=False)

    # The application is executed with recommender data now available.
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    # Loading, fitting and rendering the recommender must complete without an uncaught Streamlit exception.
    assert not app.exception

    # The page must report either a single recommended estimator or the application's approximate-tie message.
    rendered = "\n".join(str(element.value) for element in app.markdown)
    assert "Recommended estimator:" in rendered or "Approximate tie between" in rendered

    # When selector data is available, the fallback generation command must no longer be displayed.
    assert "python -m experiments.train_selector" not in [str(element.value) for element in app.code]
