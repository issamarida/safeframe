# Contributing to SafeFrame

Thanks for helping protect children's privacy in shared video. This document
describes how work flows through the repository.

## Branching model

SafeFrame uses a lightweight Git Flow:

| Branch              | Purpose                                                        | Merges into         |
| ------------------- | -------------------------------------------------------------- | ------------------- |
| `main`              | Released, tagged code only. Protected; never committed to directly. | —              |
| `develop`           | Integration branch. Every feature lands here first.            | `release/*`         |
| `feature/<topic>`   | New functionality, branched from `develop`.                    | `develop`           |
| `fix/<topic>`       | Bug fixes, branched from `develop`.                            | `develop`           |
| `chore/<topic>`     | Tooling, CI, docs, dependency updates.                         | `develop`           |
| `release/vX.Y.Z`    | Release hardening (version bump, changelog).                   | `main` + `develop`  |
| `hotfix/vX.Y.Z`     | Urgent fix to a release, branched from `main`.                 | `main` + `develop`  |

Rules:

- Merge with `--no-ff` so each feature remains visible as a unit in history.
- Every merge into `develop` or `main` goes through a pull request with green CI.
- Releases are tagged on `main` as `vX.Y.Z` ([Semantic Versioning](https://semver.org)).

Recommended GitHub branch protection for `main` and `develop`: require a pull
request, require the `CI` workflow to pass, and disallow force pushes.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org):

```
feat(pipeline): aggregate age evidence per track
fix(api): reject uploads that ffprobe cannot parse
docs: explain probabilistic age estimation
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`, `ci`, `build`.

## Development setup

```bash
make install        # backend (uv) + frontend (npm) dependencies
pre-commit install  # local guardrails
make test           # backend + frontend test suites
make dev            # API on :8000 and UI on :5173
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for the code layout, key defaults
and testing notes.

## Data handling

SafeFrame exists to protect children, so the repository is held to a strict standard:

- **Never commit footage.** No video or audio files of any kind, including
  "harmless" test clips. Tests synthesise their own video with FFmpeg at runtime.
- **Never commit model weights.** They are downloaded on first use into a cache
  outside the repository, with checksums verified where the source is pinned.
- **Never commit secrets.** Use environment variables; `.env.example` documents them.
- **Never paste footage, frames or faces into issues or pull requests.**
  Describe the scenario instead (e.g. "two people, one turns away for ~1 s").

`scripts/check_forbidden_files.py` enforces the first three rules in both the
pre-commit hook and CI.

## Pull request checklist

See [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).
Changes that affect what gets blurred must describe how they shift the balance
between missed minors (privacy failure) and over-blurring (usability cost) —
SafeFrame always errs toward blurring.
