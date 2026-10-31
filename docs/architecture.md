# ICU-EWS architecture

ICU-EWS is a single Streamlit application with a strict separation between data access, machine-learning
artifacts and presentation. The UI never trains a model, never fits a preprocessing pipeline and never
hard-codes evaluation results.

## Runtime flow

```text
app.py
  ├── ensure_state()             session defaults, dataset selection, routing
  ├── inject_css(active_nav)     design system, active navigation styling
  ├── topbar()                   brand, clock, dataset picker, global search, alerts, profile
  ├── sidebar(active_nav)        nine navigation destinations, workspace card
  └── render(view)               dispatch to icu_ews/views/<view>.py
                                     │
                                     ▼
                            repository.py (cached reads)
                                     │
                    ┌────────────────┴──────────────────┐
                    ▼                                   ▼
        artifacts.load_frames()                 artifacts.load_model_bundle()
        patients / observations / labs          models, probabilities, metrics,
                                                curves, importance, metadata
```

The application has no backend service, database server or external API. All state lives in
`st.session_state`, `st.query_params` and — for the alert workflow only — a local JSON file that is
created on first acknowledgement and removed when the workflow is cleared.

## Layers

| Layer | Modules | Responsibility |
|---|---|---|
| Entry point | `app.py` | Page configuration, shell composition, view dispatch |
| Shell | `shell.py`, `state.py` | Fixed navigation, routing, session defaults, global search |
| Design system | `theme.py`, `icons.py`, `ui/components.py` | CSS tokens, layout primitives, cards, chips, empty states |
| Charts | `ui/charts.py`, `analytics.py` | Plotly theme, calibration/ROC helpers, cohort statistics |
| Views | `views/*.py` | One workspace per module; composes components and cached data |
| Data access | `data/repository.py`, `ml/artifacts.py` | Cached dataset, snapshot, alert and SHAP access |
| Data model | `data/schema.py`, `data/demo.py`, `data/mews.py`, `data/alerts.py` | Contracts, synthetic generator, MEWS, alert rules and workflow |
| Machine learning | `ml/features.py`, `ml/training.py`, `ml/explain.py` | Feature engineering, training/evaluation, SHAP |
| Offline tooling | `scripts/` | Cohort rebuild, hyperparameter search, visual QA helpers |

## Data and feature contracts

- `data/schema.py` defines the columns and dtypes for patients, observations and laboratory panels, plus a
  `validate_bundle` function used by the bootstrap script and the test suite.
- `ml/features.py` defines the canonical 47-feature vector (`FEATURE_COLUMNS`) and metadata used for labels,
  units and grouping across the dashboard.
- `ml/artifacts.py` is the only module that reads or writes bundle files. It stores three Parquet frames,
  a joblib model bundle and JSON metadata in `artifacts/bundles/<slug>/`. On load it refits the imputer and
  scaler with the installed scikit-learn version from the stored training matrix, so bundles trained under an
  older scikit-learn remain usable after a dependency upgrade; the trained estimators are not altered.

## Leakage controls

1. The train/test split is made at the patient level with a fixed seed (`42`); no patient appears on both sides.
2. The operating threshold uses out-of-fold predictions produced by cross-validation inside the training
   partition. The held-out partition is scored once.
3. Laboratory values use the latest panel at or before the assessment anchor; the feature builder never reads
   forward in time.
4. Model selection uses cross-validated ROC-AUC on the training partition only.
5. SHAP explainers are fitted on the selected model and evaluated on the held-out features.

## Caching strategy

`data/repository.py` caches dataset frames, patient snapshots, risk timelines, alert construction and SHAP
results with `st.cache_data`. The cache key includes the dataset slug and, where relevant, the three session
thresholds, so changing a threshold recomputes bands, alerts and derived KPIs without retraining. Model bundles
are immutable for the life of the process; rebuilding a bundle requires an application restart.

## Routing and interaction model

- Routes are stored in both `st.session_state.view` and `st.query_params['view']`, which allows deep links such
  as `?view=patient&pid=ICU-4808`.
- Search, filter, sort and page-size state use namespaced keys (`ov_`, `mon_`, `an_`, `al_`, `xai_`, `pd_`,
  `px_`) so a dataset switch can reset one workspace without disturbing another.
- Buttons that navigate or reset filters are executed through callbacks, which keeps widget-backed state
  consistent between the browser and the session (Streamlit forbids mutating an instantiated widget key from
  the main script body).
- Patient cards and alert feed rows use a transparent overlay button so the whole card is clickable without
  parsing HTML clicks.

## Visual system

- A 76 px fixed top bar and a 272 px fixed sidebar frame a full-width content area.
- CSS custom properties define the navy palette, risk colours, borders, radii and shadows; every component
  consumes those tokens.
- Risk bands share one semantic colour map (`critical`, `high`, `moderate`, `low`) across cards, charts, badges
  and tables.
- Plotly figures are themed centrally in `ui/charts.py` (transparent paper, matching grid colours, shared
  tooltip style).
- The sidebar collapses to an off-canvas rail and reopens from a fixed control; at narrow widths the main area
  switches to full width.

## Failure handling

- Empty filter results render an explicit empty state rather than a blank panel.
- A missing artifact bundle falls back to the first available bundle; with no bundles the UI shows a
  configuration state.
- An unknown patient ID in a deep link renders a "not found" panel with a route back to monitoring.
- Alert workflow state is stored per alert ID; clearing the workflow removes only review records, not the
  derived alerts.

## Extension points

- Add a new cohort by writing a bundle under `artifacts/bundles/<slug>/`; the dataset picker discovers it
  automatically.
- Add a feature by extending `FEATURE_COLUMNS` and `FEATURE_META`, rebuilding the bundle and retraining.
- Add a workspace by creating a module in `icu_ews/views/`, registering it in `views/__init__.py` and adding a
  navigation entry in `config.py`.
