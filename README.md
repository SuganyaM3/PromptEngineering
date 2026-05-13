# Enterprise Prompt Engineering Studio

Production-oriented open-source GenAI training material and deployable application for prompt engineering, RAG, agents, prompt security, and multimodal AI.

## Artifacts

- `enterprise_prompt_engineering_colab.ipynb` - full Google Colab workshop notebook
- `app.py` - deployable Streamlit application
- `requirements.txt` - Python dependencies
- `Dockerfile` - container deployment
- `.streamlit/config.toml` - Streamlit theme and server config

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Models

The project uses free/open-source Hugging Face models:

- `google/flan-t5-base`
- `google/flan-t5-small`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `mistralai/Mistral-7B-Instruct-v0.2` when GPU memory is available
- `sentence-transformers/all-MiniLM-L6-v2`
- `Salesforce/blip-image-captioning-base`

## Enterprise Hardening

Before production, add authentication, authorization, DLP redaction, policy enforcement, prompt/version audit logs, retrieval source freshness checks, tool permissions, observability, and representative evaluation suites.
