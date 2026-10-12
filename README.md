

* Model Regression Detection System



A Python-based project that evaluates an LLM-powered feature against a golden dataset and detects performance regressions between baseline and candidate results.



Features



\- YAML-based versioned prompts

\- Pydantic output validation

\- JSON golden dataset with 12 test cases

\- Automated classification evaluation

\- Accuracy calculation and JSON reports

\- Baseline-versus-candidate comparison

\- Regression detection with a configurable threshold

\- Automated unit tests

\- Single-command evaluation pipeline



Tech Stack



\- Python

\- OpenAI Python SDK

\- Pydantic

\- PyYAML

\- JSON

\- unittest



Project Structure



```text

model-regression-detection/

├── app/

│   ├── classifier.py

│   ├── main.py

│   └── schemas.py

├── datasets/

│   └── golden\_dataset.json

├── evaluation/

│   ├── run\_evaluation.py

│   └── compare\_results.py

├── prompts/

│   └── support\_v1.yaml

├── reports/

├── tests/

│   └── test\_regression.py

├── .gitignore

├── requirements.txt

└── run\_pipeline.py

```



&#x20;Setup



Create and activate a virtual environment:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

python -m pip install -r requirements.txt

```



\## Run the Pipeline



```powershell

python run\_pipeline.py

```



Run automated tests separately:



```powershell

python -m unittest tests.test\_regression -v

```



Configuration



The project currently defaults to a mock classifier so the pipeline can be tested without API credits. Real-model evaluation requires a valid OpenAI API key and available API credits.



Keep secrets in `.env`. Never commit API keys to GitHub.



Project Status



The mock evaluation pipeline, baseline comparison, regression threshold, and unit tests are implemented. GitHub Actions automation and real-model evaluation are planned next.



