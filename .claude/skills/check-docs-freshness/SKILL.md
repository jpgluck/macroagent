---
Scan all documentation files for stale or incorrect claims after code changes, then fix what is wrong.

Invoke this skill after any meaningful code change: refactors, new modules, deleted files, dependency updates, architecture shifts, or renamed commands.

## Step 1 — Understand What Changed

Run `git diff HEAD` (staged + unstaged) to see what changed. If the user supplied a description of what changed, use that as additional context. Note file additions, deletions, renames, and content changes that could invalidate documentation.

```bash
git diff HEAD --stat
git diff HEAD
```

Also check recent commits if the diff is empty (e.g. changes were already committed):

```bash
git log --oneline -10
git diff HEAD~1 HEAD --stat
```

## Step 2 — Discover All Documentation Files

Glob for every documentation file in the repo:

- `CLAUDE.md` (root)
- `README.md` (root and subdirectories)
- `pyproject.toml`, `setup.cfg`, `setup.py` (metadata + deps)
- `requirements*.txt`
- Any `*.md` files in the repo root or top-level directories
- `.claude/skills/*/SKILL.md` (skill docs)

```
glob pattern: **/*.md  (limit depth to avoid node_modules/dist)
glob pattern: requirements*.txt
glob pattern: pyproject.toml
```

Read every file found. Note its path — you will cite file:line in your report.

## Step 3 — Cross-Reference Claims Against the Codebase

For each documentation file, go through every factual claim and verify it against the actual codebase. Use Grep and Glob aggressively. Do not trust the docs in isolation.

### 3a. File paths and module references

For every file path mentioned in docs (e.g. `app.py`, `utils/fred_helper.py`, `data/sample_demand.csv`, `LICENSE`):

```bash
ls <path>   # or glob to confirm existence
```

Flag any path that no longer exists. Flag any path that was renamed.

### 3b. Commands ("how to run")

For every shell command shown in docs:

- `streamlit run app.py` — confirm `app.py` exists at repo root
- `pip3 install -r requirements.txt` — confirm `requirements.txt` exists
- `python app.py`, `python -m ...` — check if this is actually the correct invocation
- Any test commands (`pytest`, `npm test`, etc.) — check if test infrastructure exists

Use Glob and Bash `ls` to verify the referenced files exist. Do not run the commands themselves.

### 3c. Test, lint, and CI status

Search for test infrastructure:

```bash
glob: tests/**/*.py
glob: test_*.py
glob: **/*.test.js
glob: .github/workflows/*.yml
glob: .pre-commit-config.yaml
glob: pyproject.toml  (check for [tool.pytest] / [tool.ruff] sections)
```

If any test files or CI configs exist, flag any doc claim that says "No tests configured", "No linter configured", "No CI", etc.

Report the actual count: e.g. "tests/ directory contains 4 files."

### 3d. Dependencies

Read `requirements.txt` (and `pyproject.toml` if present). Compare against every dependency table or list in CLAUDE.md / README.md:

- Are all listed packages still present in requirements.txt?
- Are there packages in requirements.txt not mentioned in the docs?
- Do pinned version numbers match?
- Is the Python version claim accurate? (check `.python-version`, `pyproject.toml` `requires-python`, or `runtime.txt`)

### 3e. Architecture descriptions

For every module, class, or function mentioned in docs:

```bash
grep -r "class FredHelper" .
grep -r "def rank_indicators" .
grep -r "def _validate_company_df" .
```

If a symbol was renamed or removed, flag it. If new major modules were added but are not mentioned in the architecture section, flag them.

For session state keys, compare the table in CLAUDE.md against actual `st.session_state` assignments in `app.py`.

### 3f. Feature descriptions

Read feature lists in README.md / CLAUDE.md. For each feature claimed:

- Does the code actually implement it?
- Was it removed in a recent refactor?
- Was a new feature added that is not documented?

### 3g. Broken internal links

For any `[text](path)` Markdown links that point to local files (not URLs), confirm the file exists.

## Step 4 — Report Findings as a Checklist

Print a structured report. Use checkboxes:

```
## Documentation Freshness Report

### CLAUDE.md
- [x] `streamlit run app.py` command is correct — app.py exists at root
- [x] Dependency table matches requirements.txt (all 8 packages present)
- [ ] STALE (line 42): "No tests configured" — tests/ directory exists with N test files. Update to reflect actual status.
- [ ] STALE (line 78): Lists `utils/fred_helper.py` as the only helper — new module `utils/backtest.py` added but not mentioned.

### README.md
- [x] No README.md found — nothing to check.

### requirements.txt vs docs
- [x] All packages in the CLAUDE.md table are present in requirements.txt
- [ ] STALE: `scipy==1.11.0` in requirements.txt is not listed in CLAUDE.md dependency table.

### File path checks
- [x] data/sample_demand.csv exists
- [ ] MISSING: LICENSE file referenced in README.md line 3 does not exist in repo.
```

If everything is accurate, say so clearly: "All documentation claims verified against codebase. No stale content found."

## Step 5 — Fix Stale Claims

For every item marked `[ ]` STALE or MISSING, apply a fix using the Edit tool:

- Update the stale text to match the current codebase reality
- If a file reference is broken because the file was deleted, remove or update the reference
- If a dependency was added/removed, update the table
- If architecture changed, update the description
- If "No tests configured" when tests exist, update to "Tests in `tests/` — run with `pytest`" (or whatever is accurate)
- Do NOT remove compliance disclaimers or legal notices
- Do NOT change the meaning of a claim speculatively — only fix what you can verify

After all edits, re-read each modified file to confirm the fix is correctly applied.

## Step 6 — Summary

Print a one-paragraph summary of what was checked, what was stale, and what was fixed. Include counts:

> "Checked 3 documentation files. Found 4 stale claims: 2 in CLAUDE.md, 1 in README.md, 1 in requirements mismatch. Fixed all 4. No broken file links remain."

If nothing was stale:

> "Checked 3 documentation files across N claims. All accurate against current codebase. No edits needed."

## Efficiency Notes

- Glob and Grep broadly before reading — avoid reading large files just to find one symbol
- Run independent checks in parallel where possible (e.g. glob for test files while reading requirements.txt)
- When verifying a function exists, prefer `grep -r "def function_name"` over reading the whole file
- Cite specific file:line for every finding so the user can navigate directly to it
- Never fabricate findings — only report what you can positively verify or refute
