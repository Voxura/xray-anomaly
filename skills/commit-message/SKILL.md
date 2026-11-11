---
name: commit-message
description: >-
  Enforces the user's specific Conventional Commit style: `<type>(<scope>): <change description>`.
  Use whenever composing, proposing, amending, or executing git commits across any project or repository.
---

# Commit Message Style Guide

This skill defines the user's exact commit message format, conventions, and quality standards for all git commits.

---

## The Core Format

All commit subject lines must strictly adhere to:

```text
<type>(<scope>): <change description>
```

### Components

1. **`type`**: The category of change (lowercase).
2. **`scope`**: The specific component, subsystem, module, or area touched (lowercase, in parentheses).
3. **`change description`**: The exact change applied, written in the imperative mood.

---

## Critical Rules & Guidelines

### 1. No Redundancy Between Scope and Description
- The scope `(<scope>)` already specifies **what** component or feature was touched.
- The description must strictly explain **what changed**, without repeating the name of the file, scope, or feature.
- **Incorrect:**
  - ❌ `fix(blank.ico): fix blank.ico bug`
  - ❌ `feat(tabs): tabs feature added`
  - ❌ `feat(license): add license to license file`
- **Correct:**
  - ✅ `fix(assets): restore 32-bit ARGB multi-resolution blank icon`
  - ✅ `feat(tabs): freeze tab positions on middle-click until mouse exit, clamp plus button at limit`
  - ✅ `feat(license): add MIT license`

### 2. Supported Commit Types

| Type | When to Use | Examples |
| :--- | :--- | :--- |
| **`feat`** | New feature, capability, or user-facing functionality | `feat(core): add cli flags`<br>`feat(logo): add svg logo`<br>`feat(tabs): implement Brave progressive shrinking with title elision` |
| **`fix`** | Bug fix or issue resolution | `fix(core): resolve Windows 11 black icon overlay and registry path`<br>`fix(core): fix unelevated file explorer restart` |
| **`refactor`** | Code/build restructuring without altering behavior or adding features | `refactor(build): optimize compilation pipeline with object library, PCH, -O2`<br>`refactor(tabs): migrate tab strip to declarative horizontal ListView` |
| **`build`** | Build system, toolchain, dependency, or CI/workflow updates *(can also be scoped under `feat(build):` or `refactor(build):`)* | `feat(build): speed up mingw install and add build verification footer`<br>`refactor(build): optimize compilation pipeline` |
| **`chore`** | Maintenance, house-cleaning, dependency bumps, or tool configurations | `chore(deps): bump actions/checkout to v4`<br>`chore(core): consolidate files into appropriate folders` |
| **`docs`** | Documentation or README changes | `feat(readme): update storage usage in the intro`<br>`docs(api): document ltx parser parameters` |
| **`perf`** | Performance-specific optimizations | `perf(vfs): reduce heap allocations during archive read` |
| **`style`** | Code formatting, whitespace, or linting without behavioral change | `style(ui): clean up indentation in CustomTitleBar` |

### 3. Casing & Phrasing
- Subject line starts with lowercase `<type>(<scope>): ` followed by a lowercase, imperative action verb.
- Do not append a trailing period (`.`) to the subject line.
- Be concise yet descriptive: state the precise technical mechanism or user-visible outcome.

---

## Multi-Line & Detailed Commits

For complex, multi-faceted changes or significant bug fixes, provide a high-level subject line followed by a blank line and concise, technical bullet points:

```gitcommit
fix(core): resolve Windows 11 black icon overlay and registry path

- Remove trailing ',0' resource index suffix from Shell Icons registry value 29, which caused Windows 11 shell image loaders to fail with ERROR_FILE_NOT_FOUND
- Replace blank.ico with a valid 32bpp multi-resolution icon (16x16 through 256x256) with proper alpha channels
- Extract blank.ico to %SystemRoot%\blank.ico (C:\Windows\blank.ico) with automatic fallback to %ProgramData%\DeLNK\blank.ico
- Update cleanup routines to remove both Windows and ProgramData icon locations
```

### Rules for the Body:
- Separate subject and body with a single blank line.
- Use clean bullet points starting with `- `.
- Highlight the **root cause**, **exact technical solution**, and **fallbacks/cleanup**.
- Keep bullets factual, technical, and free of fluff.

---

## Commit Workflow

1. **Inspect changes**: Always inspect staged changes (`git diff --staged` or `git status`).
2. **Draft the message**: Formulate `<type>(<scope>): <change description>`.
3. **Verify against redundancy**: Confirm that the description does not redundantly repeat the scope name.
4. **Propose before committing**: When in an interactive pair programming session or when committing non-trivial work, show the drafted commit message to the user for review before committing.
