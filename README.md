# Evidence Ledger

Evidence Ledger is a research prototype for portfolio-based diagnosis and personalized professional development for higher education faculty. This release contains the local Web application and the M1 method evaluation for RQ1.

## Data provenance

M1 uses expert-constructed faculty portfolio cases and evaluation materials. The dataset is organized around predefined teacher profiles, professional evidence, competency indicators, and controlled portfolio variations. The released construction scripts use seeded templates to assemble materials, dates, contributor roles, and evidence perturbations consistently across evaluation conditions.

The final reference labels and blinded development path ratings are obtained through repeated `qwen3.7-flash` evaluations. The annotation pipeline uses three isolated rounds and a separate adjudication pass for qualifying disagreements. Development construction labels serve as training references. The automated tests use local fixtures in place of external model calls.

## Run the application

From the release root, use Python 3.11 or newer. The Web application uses the Python standard library. PDF extraction can use Poppler's `pdftotext` or the optional `pypdf` package. Set the model provider's OpenAI-compatible API base URL and API key through your shell or process manager. The app also accepts `V_API_BASE_URL` and `V_API_KEY`.

```bash
export EVIDENCE_LEDGER_LLM_BASE_URL=https://your-provider.example/v1
export EVIDENCE_LEDGER_LLM_MODEL=qwen3.7-flash
python3 -m app.server --host 127.0.0.1 --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). Set `EVIDENCE_LEDGER_LLM_API_KEY` to enable diagnosis and path generation. The UI and resource catalog are available with an empty key. The app stores each workspace in a local SQLite database under `runtime/`. The configuration template lists storage, request limits, and proxy settings. [Environment configuration](.env.example)

1. Save the faculty profile, development goal, and weekly time budget.
2. Register materials with a source identifier, date, and contribution level. The app extracts text and keeps source locations. It supports PDF, DOCX, PPTX, TXT, Markdown, CSV, JSON, and HTML.
3. Build the evidence map. The model proposes quotations, and the system checks them against the source text and groups copies of the same work by source. The evidence map shows the decision and its traceable evidence for each indicator.
4. Review the action for each indicator. Assessable evidence can support a rating from 1 to 4. For a matching goal, ratings 1–2 lead to targeted practice and ratings 3–4 lead to extension and integration. Insufficient evidence prompts a request for specific additional material. Conflicting evidence goes to the local human-review queue.
5. Open the development path to see the learning objective, practice task, artifact to retain, and supporting resources. Recommendations draw on the bundled resource catalog and its accessibility, prerequisite, and time metadata.

The server uses provisional rating rules by default. To use the frozen M1 calibration, start the app with `EVIDENCE_LEDGER_CALIBRATION_FILE=method_validation/data/expert_ratability_calibration.json`. This artifact fits the M1 calibration split and its model-adjudicated reference labels. The `expert_calibrated` field is the runtime schema flag for a versioned calibration artifact. [Calibration artifact](method_validation/data/expert_ratability_calibration.json)

## Reproduce the M1 evaluation

The frozen M1 evaluation uses Python 3.12.3 and NumPy 2.5.5. The bundled expert-constructed portfolio materials, reference labels, predictions, and blinded path ratings allow the reported analysis to be recomputed offline. The command verifies the frozen files and data boundaries, regenerates summary statistics and figures under `method_validation/results/`, and runs the tests. Use a disposable copy to preserve the distributed results as a comparison snapshot.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r method_validation/analysis/requirements_validation.txt
bash method_validation/analysis/reproduce.sh
```

To check the frozen files and input boundaries without regenerating results, run `python3 method_validation/analysis/verify_integrity.py`.

To rerun the evaluation pipeline with model access, set `V_API_KEY` and `V_API_BASE_URL` for an endpoint serving `qwen3.7-flash`, then run `bash method_validation/analysis/reproduce.sh --with-model` in a disposable copy. The scripts reuse compatible cached responses. Fresh responses can change labels, predictions, and scores. The pipeline uses the development split for B1 and B2 and the calibration split for the B3 confidence threshold and P rating eligibility. The paired main test uses the same test inputs across all methods. B3, P, and both ablations share three model readings per record.

| Split | Teacher cases | Portfolio records | Inputs and references |
| --- | ---: | ---: | --- |
| Development | 36 | 216 | `development_records.jsonl`, `development_reference.jsonl` |
| Calibration | 12 | 72 | `calibration_records.jsonl`, `calibration_reference.jsonl` |
| Main test | 48 | 288 | `final_test_records.jsonl`, `final_reference.jsonl` |

The six main-test versions are the original portfolio (C0), key-evidence removal (T-del), a same-source addition (T-dup), an irrelevant addition (T-irr), material reordering (T-order), and independent current counterevidence (T-conflict). These controlled variations are applied to the expert-constructed portfolio cases to evaluate how the methods respond to changes in evidence availability, relevance, ordering, and consistency.

Teacher cases are disjoint across splits. Keep condition labels, case IDs, and reference judgments out of predictors when using the released benchmark. The auxiliary files cover 384 C0 indicator records and 240 nontarget-indicator perturbation records. M1 inputs and rubric content retain their original language. [M1 data](method_validation/data/) · [Frozen manifest](method_validation/data/frozen_manifest.json)

| Manuscript result | Source artifact under `method_validation/results/` |
| --- | --- |
| RQ1 primary comparison, Figure 5, and Table S1 | `statistical_tests.json`, `metrics.json`, `action_confusion_*.csv` |
| Conditional ratings, evidence localization, and path evaluation, Table S2 | `metrics.json`, `path_ratings_summary.json` |
| Paired secondary comparisons, Table S3 | `statistical_tests.json` |
| Stability and nontarget indicators, Table S4 | `metrics.json`, `locality_metrics.json`, `table_M2.csv` |
| Component ablations, Table S5 | `metrics.json`, `table_M1.csv`, `table_M2.csv`, `state_confusion_P_minus_Family.csv` |
| C0 across all indicators and annotation agreement, Table S6 | `c0_all_indicator_metrics.json`, `annotation_agreement.json` |
| M1 plots underlying Supplementary Figures S1–S3 | `state_confusion_*.csv`, `risk_coverage.csv`, `metrics.json` |

The primary endpoint compares erroneous remediation on the 48 T-del records. P produced 0 errors and B3 produced 21. The Jeffreys-adjusted difference was −42.9 percentage points. Inference pairs outcomes by teacher case and uses 10,000 bootstrap resamples alongside the exact McNemar test. [Primary comparison](method_validation/results/statistical_tests.json)

## Repository layout

| Path | Contents |
| --- | --- |
| `app/` | HTTP application, Web UI, source-text checks, evidence grouping, evidence-state decisions, and development paths |
| `app/data/` | Competency framework, resource catalog, and calibration schema |
| `method_validation/data/` | Expert-constructed M1 portfolio materials, references, prompts, calibration and training artifacts, raw annotations, and frozen checksums |
| `method_validation/predictions/` | Frozen B1, B2, B3, P, ablations, shared model reads, and blinded path ratings |
| `method_validation/analysis/` | Portfolio construction, annotation, prediction, analysis, and verification tools |
| `method_validation/results/` | M1 statistics, tables, and plots |
| `tests/` | Application, input handling, and annotation cache tests |

## Licensing

License selection is pending for the software and the M1 benchmark.