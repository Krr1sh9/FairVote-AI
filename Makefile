.PHONY: install-dev install-locked sync-lock lock verify-lock lint type test ci evidence-smoke examiner-reproduce reproduce-examiner final-evidence reproduce-final verify-final robustness theory bundle publication report privacy-evidence clean

PYTHON ?= python
OUT ?= evidence/final

LOCK ?= requirements.lock.txt
UV_PYTHON_VERSION ?= 3.14
UV_PLATFORM ?= x86_64-manylinux_2_28

install-locked:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(LOCK)
	$(PYTHON) -m pip install -e . --no-deps

sync-lock:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(LOCK)

lock:
	uv pip compile pyproject.toml --all-extras --python-version $(UV_PYTHON_VERSION) --python-platform $(UV_PLATFORM) --output-file $(LOCK)

verify-lock:
	$(PYTHON) -m scripts.verify_lockfile --lock $(LOCK) --pyproject pyproject.toml

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev,browser]"

lint:
	$(PYTHON) -m compileall -q fairvote app respondent experiments tests
	ruff format --check .
	ruff check .

type:
	mypy fairvote app respondent experiments

test:
	$(PYTHON) -m pytest -q --cov=fairvote --cov=experiments --cov=respondent --cov=app --cov-report=term-missing:skip-covered --cov-fail-under=60

ci: lint type test evidence-smoke theory

evidence-smoke:
	$(PYTHON) -m experiments.mrp_vs_baselines --preset smoke_test --disable_neural --out_dir /tmp/fairvote-smoke --fail_fast --skip_plots

examiner-reproduce:
	$(PYTHON) -m experiments.mrp_vs_baselines --preset examiner_reproduce --disable_neural --out_dir $(OUT) --fail_fast --skip_plots

reproduce-examiner: verify-lock examiner-reproduce theory
	$(PYTHON) -m experiments.build_results_bundle --root $(OUT)

robustness:
	$(PYTHON) -m experiments.mrp_vs_baselines --preset robustness_evidence --out_dir $(OUT) --fail_fast

final-evidence:
	$(PYTHON) -m experiments.mrp_vs_baselines --preset final_evidence --out_dir $(OUT) --fail_fast

reproduce-final: verify-lock final-evidence theory bundle
	@echo "Set RUN to the generated evidence/final/<RUN_DIR> and run make publication verify-final RUN=$$RUN"

verify-final:
	@test -n "$(RUN)" || (echo "Set RUN=evidence/final/<RUN_DIR>" && exit 2)
	$(PYTHON) -m scripts.verify_final_evidence --run_dir $(RUN) --require_paired

theory:
	$(PYTHON) -m experiments.theory_validation --out_dir $(OUT)/theory --quick

bundle:
	$(PYTHON) -m experiments.build_results_bundle --root $(OUT)

publication:
	@test -n "$(RUN)" || (echo "Set RUN=evidence/final/<RUN_DIR>" && exit 2)
	$(PYTHON) -m experiments.write_publication_result --run_dir $(RUN) --out_dir $(RUN)
	$(PYTHON) -m experiments.write_publication_result --run_dir $(RUN) --out_dir paper/generated

report: publication

privacy-evidence:
	$(PYTHON) -m scripts.verify_privacy_evidence

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name htmlcov \) -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete
