# Deployment notes

ICU-EWS is a self-contained Streamlit application. It needs no database server, message queue or external API.
The trained artifact bundle is committed, so a fresh clone can serve the dashboard without retraining.

> ICU-EWS is a research prototype. Any public or shared deployment must keep the demonstration-data labelling and
> must not be presented as a clinical system.

## Requirements

- Python 3.11 or newer
- The packages in `requirements.txt`
- About 500 MB of free memory for the Streamlit process with the demo bundle loaded

## Local deployment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.headless true --server.port 8501
```

The application is available at `http://localhost:8501`. The bundled `.streamlit/config.toml` sets the dark
theme, disables the Streamlit toolbar navigation and turns off usage statistics collection.

### Production-style process

For a long-running local or internal deployment, run Streamlit under a process supervisor (systemd, launchd or a
container platform of your choice). The command below is the same entry point used during development:

```bash
streamlit run app.py \
  --server.headless true \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.fileWatcherType none
```

`fileWatcherType none` is recommended when source files do not change at runtime; restart the process after a code
change.

## Streamlit Community Cloud

The repository is public and contains `requirements.txt` and `app.py` at the root, which is the layout Streamlit
Community Cloud expects:

1. Create a new app from the repository.
2. Select branch `main` and entry point `app.py`.
3. Deploy.

No secrets are required. The demo artifact bundle is loaded from the repository; training is not executed at
startup.

## Data and state

- Cohort data and model artifacts are read from `artifacts/bundles/` and treated as read-only.
- The alert acknowledgement/resolution workflow writes `artifacts/alert_state.json` locally. That file is
  git-ignored and is recreated on demand. On ephemeral hosting it resets when the container is replaced; this is
  expected for a demo workspace.
- Rebuilding a bundle with `scripts/bootstrap_demo.py` changes on-disk artifacts. Commit the regenerated bundle
  deliberately rather than letting a host regenerate it at startup.

## Security and privacy

- No credentials, tokens or patient data are required or stored by the application.
- `.env`, local secrets and `data/raw/` are git-ignored.
- The UI never sends alerts or records to an external system. If you add an integration, treat it as a new
  security and clinical-safety boundary and review it before deployment.
- Restrict access if the deployment is visible outside your team; the demo data is synthetic but the interface is
  still labelled and intended for research use only.

## Regenerating artifacts

```bash
python scripts/bootstrap_demo.py --seed 7 --patients 420 --slug demo-cohort-v1
```

This rewrites `artifacts/bundles/demo-cohort-v1/` (models, metrics and Parquet frames). Verify the held-out
metrics printed by the script before committing a regenerated bundle.
