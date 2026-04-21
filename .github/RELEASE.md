# Release Checklist

Step-by-step guide for publishing a new version of `cognitive-scoring` to PyPI.

## Prerequisites (one-time)

1. **Configure PyPI Trusted Publisher** at
   [pypi.org/manage/project/cognitive-scoring/settings/publishing/](https://pypi.org/manage/project/cognitive-scoring/settings/publishing/):
   - Owner: `suncloudsmoon`
   - Repository: `cognitive-scoring`
   - Workflow: `release.yml`
   - Environment: `pypi`

2. **Create the `pypi` environment** in your GitHub repository at
   `Settings → Environments → New environment → pypi`. Optionally add
   required reviewers for manual approval before publishing.

## Cutting a Release

### 1. Update the version

Bump the version in `pyproject.toml`:

```toml
[project]
version = "1.1.0"  # ← update this
```

> **Note:** `tribev2/__init__.py` reads the version dynamically from package
> metadata — you do NOT need to update it separately.

### 2. Update the changelog

Add a new section to `CHANGELOG.md`:

```markdown
## [1.1.0] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

Update the comparison links at the bottom of the file:

```markdown
[Unreleased]: https://github.com/suncloudsmoon/cognitive-scoring/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/suncloudsmoon/cognitive-scoring/compare/v1.0.0...v1.1.0
```

### 3. Commit, tag, and push

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "release: v1.1.0"
git tag v1.1.0
git push origin main --tags
```

Pushing the `v1.1.0` tag automatically triggers `release.yml`, which builds
the distribution and publishes to PyPI via OIDC. No GitHub Release required.

### 4. (Optional) Create GitHub Release for visibility

After PyPI publishing succeeds, you can create a GitHub Release from the tag
so the changelog appears on the repo's Releases page. This is purely for UX
and does not trigger anything.

### 5. Verify

- [ ] GitHub Actions workflow completes with green ✅
- [ ] Package appears at [pypi.org/project/cognitive-scoring/](https://pypi.org/project/cognitive-scoring/)
- [ ] `pip install cognitive-scoring==1.1.0` works
- [ ] `python -c "import tribev2; print(tribev2.__version__)"` prints `1.1.0`

## Troubleshooting

### "Tag does not match pyproject.toml version"

The `check` job validates that the git tag `v1.1.0` matches the version
`1.1.0` in `pyproject.toml`. If they don't match, fix the version and
re-tag:

```bash
git tag -d v1.1.0              # delete local tag
git push origin :v1.1.0        # delete remote tag
# fix pyproject.toml, commit, then re-tag
git tag v1.1.0
git push origin main --tags
```

### "OIDC token exchange failed"

This means the Trusted Publisher is not configured, or the environment name
doesn't match. Double check:
- The PyPI Trusted Publisher settings match exactly (owner, repo, workflow, environment)
- The GitHub environment `pypi` exists in your repository settings

### Build succeeds but publish is skipped

The publish job only runs when a tag is pushed (`refs/tags/v*`). If you
triggered the workflow via **Run workflow** from the Actions tab without
pushing a tag, the build runs in debug mode and publishing is skipped by
design. Push a `v*` tag to publish.
