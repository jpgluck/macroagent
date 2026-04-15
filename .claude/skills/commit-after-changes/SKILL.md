---
name: Prompt for commit after successful changes
description: After each successful code change or feature, suggest a concise commit message and ask if user wants to commit, rather than waiting until session end
---

## When to Use

After you successfully complete a code change that:
- Passes basic checks (no syntax errors, runs without crashing)
- Adds a feature, fixes a bug, or refactors a section
- Is logically complete and testable

Do NOT wait until the end of the session. Suggest a commit immediately.

## What to Do

1. **Verify the change works** — run tests, check for errors, confirm expected behavior
2. **Suggest a commit message** — short, imperative form, describe the *why* not just the *what*
   - Good: `fix: handle NaN in pct_change before correlation`
   - Good: `feat: add Consumer Credit indicator to catalogue`
   - Bad: `update code` or `changes made`
3. **Ask for confirmation** — "Ready to commit? (y/n)" or similar
4. **Execute if confirmed** — run `git add`, `git commit -m "..."`, `git status`

## Commit Message Format

Follow conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code reorganization without behavior change
- `docs:` — documentation only
- `test:` — test additions/changes
- `chore:` — build, deps, CI/CD

One-liner (under 72 chars), then optionally a blank line + details.

## Why This Matters

- Keeps work incremental and committable instead of accumulating changes over hours
- Creates natural checkpoints so long sessions don't result in lost work
- Builds the habit of atomic commits that are easy to revert if needed
- Makes git history readable and useful for future debugging
