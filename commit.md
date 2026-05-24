# Commit Message Convention

## Format

```
<type>(<optional scope>): <description>

<optional body>

<optional footer>
```

## Special Commits

| Situation | Format |
|-----------|--------|
| Initial commit | `chore: init` |
| Merge commit | `Merge branch '<branch-name>'` (default git message) |
| Revert commit | `Revert "<reverted commit subject line>"` (default git revert message) |

## Types

| Type | When to use |
|------|-------------|
| `feat` | Add, adjust, or remove a feature |
| `fix` | Fix a bug introduced by a preceding `feat` commit |
| `refactor` | Rewrite or restructure code without changing behavior |
| `perf` | Refactor specifically aimed at improving performance |
| `style` | Code style only (whitespace, formatting, missing semicolons) |
| `test` | Add missing tests or correct existing ones |
| `docs` | Documentation changes only |
| `build` | Build tools, dependencies, project version |
| `ops` | Infrastructure, deployment, CI/CD, monitoring, backups |
| `chore` | Miscellaneous tasks (e.g., init commit, modifying `.gitignore`) |

## Scope

- Optional — provides additional context
- Project-defined (e.g., `api`, `model`, `index`, `serving`)
- Do **not** use issue identifiers as scopes

## Breaking Changes

Indicate with `!` before the `:` in the subject line:

```
feat(api)!: remove status endpoint
```

Describe the breaking change in the footer:

```
BREAKING CHANGE: ticket endpoints no longer support listing all entities.
```

## Description Rules

- Mandatory
- Use imperative, present tense: `change` not `changed` or `changes`
- Think: *"This commit will..."*
- Do not capitalize the first letter
- Do not end with a period

## Body

- Optional
- Use imperative, present tense
- Explain the motivation and contrast with previous behavior

## Footer

- Optional, **required** if breaking changes are introduced
- Reference issues: `Closes #123`, `Fixes JIRA-456`
- `BREAKING CHANGE:` followed by a space (single-line) or two newlines (multi-line)

## Versioning

| Commit contains | Version bump |
|-----------------|-------------|
| Breaking changes | Major (`1.0.0 → 2.0.0`) |
| `feat` or `fix` | Minor (`1.0.0 → 1.1.0`) |
| Anything else | Patch (`1.0.0 → 1.0.1`) |

## Examples

```
feat: add email notifications on new direct messages

feat(shopping-cart): add the amazing button

feat!: remove ticket list endpoint

refers to JIRA-1337

BREAKING CHANGE: ticket endpoints no longer support listing all entities.

fix(shopping-cart): prevent ordering an empty cart

fix(api): fix wrong calculation of request body checksum

fix: add missing parameter to service call

The error occurred due to <reasons>.

perf: decrease memory footprint for unique visitors using HyperLogLog

build: update dependencies

build(release): bump version to 1.0.0

refactor: implement fibonacci as recursion

style: remove empty line
```
