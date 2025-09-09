# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-09-09

- Docs: Expanded README (config, CLI, HTTP API + OpenAPI, Docker, evals metrics).
- Feat: Added usage metrics to predictions (lm_calls, tool_calls) and aggregated in evals.
- Feat: Added Dockerfile and Make targets (docker-build/run).
- Feat: Server endpoints `/health`, `/healthz`, `/version`; CLI flags `--func-calls` / `--no-func-calls`.
- Feat: Evals now support `expect_key` and print `contains_hit_rate` and `key_hit_rate`.
- Feat: Optional extras `[repair]` to enable json-repair parsing for decisions.
- Fix: Ensure math/time tools are present in trace when OpenAI function-calls path returns finalize prematurely.
- Chore: Silence Python 3.13 AST deprecation warnings in calculator.
- CI: Added GitHub Actions to run tests and a small eval smoke on pushes/PRs.

## [0.1.0] - 2025-09-08

- Initial release with DSPy-based micro agent, tools, CLI, FastAPI server, and basic evals.

[0.1.1]: https://github.com/evalops/dspy-micro-agent/compare/v0.1.0...v0.1.1
