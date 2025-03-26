![Querymancer interface](.github/banner.png)

# Querymancer

AI agent that lets you to talk to your database using natural language. All locally.

Features:

- Natural language to SQL query conversion
- Database schema introspection
- Context-aware conversation
- Multi-model inference with:
  - **Groq** (`llama-3.3-70b-versatile`) for general queries
  - **SambaNova** (`DeepSeek-R1`) for complex analytical queries
- Advanced optimization strategies:
  - Dynamic complexity routing
  - Token optimization pipeline
  - Parameter auto-tuning

Read the full tutorial on MLExpert.io: [mlexpert.io/v2-bootcamp/build-ai-agent](https://mlexpert.io/v2-bootcamp/build-ai-agent)

## Install

Make sure you have [`uv` installed](https://docs.astral.sh/uv/getting-started/installation/).

Clone the repository:

```bash
git clone git@github.com:mlexpertio/querymancer.git .
cd querymancer
```

Install Python:

```bash
uv python install 3.12.8
```

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

Install package in editable mode:

```bash
uv pip install -e .
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

### Create SQLite database

You can use any SQLlite database. This project comes with a sample script that can create one for you:

```sh
bin/create-database
```

This should create a file called `ecommerce.sqlite` in the `data` directory. Here's a diagram of the database schema:

![SQLite database schema](.github/db-schema.png)

### API Keys

Querymancer now uses cloud-based LLM services for inference. You'll need API keys to use these services:

1. **Groq API** - Get your API key from https://console.groq.com/keys
2. **SambaNova API** - Get your API key from SambaNova's website

Rename the `.env.example` file to `.env` and add your API keys inside:

```bash
mv .env.example .env
```

Your `.env` file should contain:
```
GROQ_API_KEY=your_groq_api_key_here
SAMBANOVA_API_KEY=your_sambanova_api_key_here
```

Look into the [`config.py`](querymancer/config.py) file to set your preferred model.

## Run the Streamlit app

Run the app:

```bash
streamlit run app.py