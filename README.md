# 


## Prerequisites

- Python 3.8+ installed

## Install (recommended)

1. Clone the repo (or download):
   ```bash
   git clone https://github.com/Mayank2177/RID_mayank.git
   cd RID_mayank
   ```
2. Create and activate a virtual environment (recommended):
   - macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```bash
   # If a requirements.txt exists
   pip install -r requirements.txt

   # Otherwise at minimum install Streamlit
   pip install streamlit
   ```

## Run the app

From the project root (where `app.py` is located) run:

```bash
streamlit run app.py
```

By default Streamlit serves at http://localhost:8501. To use a specific port (e.g. 8080):

```bash
streamlit run app.py --server.port 8080
```

If you want to run headless (useful on servers):

```bash
streamlit run app.py --server.headless true --server.port 8501
```

## Troubleshooting

- If Streamlit is not found, ensure your virtual environment is activated and `streamlit` is installed.
- If dependencies are not listed, inspect `app.py` for imports and add any missing packages to `requirements.txt`.

  pip freeze > requirements.txt
  ```
- For deployment (Heroku, Docker, Streamlit Cloud), follow the respective platform instructions and ensure `requirements.txt` and `app.py` are present.

---

