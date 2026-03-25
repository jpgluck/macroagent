---
Before making any changes, check all dependencies in package.json (or requirements.txt) for compatibility with [runtime/SDK version]. List any conflicts and propose an upgrade plan. Do NOT install anything yet.

Steps:
1. Detect the package manifest (requirements.txt or package.json) and read it.
2. Get the current runtime version (python3 --version or node --version).
3. For each dependency, check: pinned version, Requires-Python / engines field, and latest available version.
4. Identify any version gaps (patch, minor, major) and flag known breaking changes.
5. Output a table: package | pinned | latest | gap | risk.
6. Propose a tiered upgrade plan (safe now / test first / hold) with reasons.
7. Do NOT run pip install, npm install, or any package-modifying command.
---
