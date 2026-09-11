# Third-party licenses

This project's own code is MIT-licensed (see `LICENSE`). It depends on the
packages below, none of which impose any obligation beyond what pip/uv
installation already handles — nothing here is vendored or redistributed,
only depended on. Checked directly against each package's own installed
metadata (`License` / `License-Expression`), not assumed from a package
index page.

## Direct dependencies

| package | license |
|---|---|
| [`langchain-core`](https://pypi.org/project/langchain-core/) | MIT |
| [`langchain-deepseek`](https://pypi.org/project/langchain-deepseek/) | MIT |
| [`langgraph`](https://pypi.org/project/langgraph/) | MIT |
| [`httpx`](https://pypi.org/project/httpx/) | BSD-3-Clause |
| [`opentelemetry-sdk`](https://pypi.org/project/opentelemetry-sdk/) | Apache-2.0 |
| [`opentelemetry-api`](https://pypi.org/project/opentelemetry-api/) | Apache-2.0 |
| [`opentelemetry-semantic-conventions`](https://pypi.org/project/opentelemetry-semantic-conventions/) | Apache-2.0 |

## Notable transitive dependencies

Pulled in by the above, not declared directly — checked because they're
load-bearing, not because every transitive package was audited.

| package | license |
|---|---|
| `langgraph-prebuilt` | MIT |
| `langgraph-checkpoint` | MIT |
| `langgraph-sdk` | MIT |
| `pydantic` / `pydantic-core` | MIT |

## Development-only tools

Used to build and gate this project (`just gate`); never installed by, or
shipped to, anyone just running the chapters.

| package | license |
|---|---|
| `mypy` | MIT |
| `pytest` | MIT |
| `ruff` | MIT |
| `basedpyright` | MIT |

`basedpyright` bundles a Node.js runtime (`nodejs-wheel-binaries`) to run
its underlying `pyright` engine — a dev-tooling detail, not something this
project's own code touches or redistributes.

## Compatibility

MIT, BSD-3-Clause, and Apache-2.0 are all permissive licenses fully
compatible with this project's own MIT license. Apache-2.0's NOTICE-file
requirement applies only to projects that vendor or redistribute Apache-
licensed source directly; this project depends on those packages through
normal package installation and vendors none of them.
