# Incidents

Incidents is a LangGraph CLI for triaging sample production alerts. A router selects logs, deploys, and metrics specialists to investigate in parallel. An analyst combines their findings into a diagnosis and proposes remediation. Policy gates high-risk actions for team approval before publishing a Markdown runbook.

## Setup

Install [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/). Start Ollama and run `ollama signin` to use the default cloud model.

```sh
git clone https://github.com/vishal24p/Incidents.git
cd Incidents
uv sync --locked
```

Copy `.env.example` to `.env` with `Copy-Item .env.example .env` in PowerShell or `cp .env.example .env` on macOS or Linux. The defaults use `beam:cloud` via local Ollama. For another OpenAI-compatible provider, set `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` in `.env`. Use a model that supports tool calling.

## Run an incident

```sh
uv run --locked python app.py start --thread checkout-demo --alert INC-2041
```

Use a unique thread ID for each run. `INC-2041` is a checkout alert and `INC-2042` is a search latency alert. Both use synthetic data from `data/`.

If the workflow pauses, use a team listed in the output and repeat for any other pending teams. For example, approve a pending Payments action with:

```sh
uv run --locked python app.py resume --thread checkout-demo --team payments --decision approve
```

Use `--decision reject` to reject the team's pending actions. To retry a failed investigation, resume with only `--thread`. Checkpoints live in `triage_checkpoints.db`. Published runbooks appear under `runbooks/` and include approved actions and actions that pass policy without review.

`status_page.py` contains the status page scaffold. Loading incident details is pending.

## Offline tests

```sh
uv run --locked python -m unittest discover -s tests -v
```
