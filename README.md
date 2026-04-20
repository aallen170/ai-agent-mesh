# ai-agent-mesh

## Local setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows (Git Bash)
   pip install -e ".[dev]"
   ```

2. Install the git hooks (enforces ticket numbers in commit messages):
   ```bash
   git config core.hooksPath scripts/hooks
   ```

3. Start Redis:
   ```bash
   docker compose -f infra/docker-compose.yml up -d
   ```

4. Verify everything is working:
   ```bash
   python scripts/smoke_test_redis.py
   python scripts/smoke_test_registry.py
   ```