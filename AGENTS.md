# Project Agent Instructions

## Objective

Build a production-oriented Python platform that maximizes the useful coverage of every Tushare Pro points-based interface available to a 15,000-point account. Every discovered interface must have an explicit lifecycle state, collection policy, quality contract, and downstream information product or documented archival purpose.

## Engineering rules

- Use Python 3.11 or newer and `uv` for dependency management and command execution.
- Keep responsibilities separated. Put reusable logic in focused packages instead of large orchestration modules.
- Use complete, readable variable and method names. Avoid abbreviations that hide domain meaning.
- Every function must have a detailed docstring describing behavior and at least one input/output example.
- Every test case must document its scenario and expected result.
- Cover every function with tests and target 100% coverage.
- Tests must run without a real Tushare token or network access. Live probes require an explicit opt-in flag.
- Never persist, print, hash into cache keys, or include a Tushare token in exceptions, logs, manifests, or fixtures.
- Production Tushare transport must require HTTPS. Do not silently use an SDK path that sends credentials over plain HTTP.
- Preserve raw responses before normalization. Keep source units and fields auditable.
- Treat announcement time, effective time, observation time, and report period as separate concepts. Point-in-time behavior must fail closed.
- Separate points-based interfaces, independently purchased interfaces, deprecated interfaces, and empirically unavailable interfaces.
- Do not claim an interface is covered until its catalog policy, ingestion path, quality gate, tests, and information consumer are all present.

## Delivery rules

- Implement changes on a feature branch and create a Merge Request targeting `main`.
- Do not merge, deploy, submit approvals, or configure production schedules unless the user explicitly requests it.
- Commit only source code, tests, documentation, and deterministic fixtures. Never commit credentials or live provider payloads containing sensitive data.
