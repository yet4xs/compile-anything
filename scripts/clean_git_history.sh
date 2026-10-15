#!/usr/bin/env bash
# Prepared history cleanup for the leaked SSH password — DO NOT RUN YET.
#
# Preconditions (in this order, per the incident remediation plan):
#   1. The server owner has ROTATED the credential via the provider
#      console (rotation makes the leaked string worthless; rewriting
#      history first does NOT).
#   2. The owner has explicitly confirmed the force-push: this rewrites
#      public history on origin/main; every clone becomes divergent.
#
# The leaked secret is passed via environment variable so this script
# itself never contains it.
#
# Usage:
#   LEAKED_SECRET='<old password>' bash scripts/clean_git_history.sh
set -euo pipefail

: "${LEAKED_SECRET:?set LEAKED_SECRET to the old (rotated) password}"

command -v git-filter-repo >/dev/null 2>&1 || {
    echo "install git-filter-repo first: pip install git-filter-repo" >&2
    exit 1
}

# Safety: refuse to run on a dirty tree or outside the repo root
[ -z "$(git status --porcelain)" ] || { echo "clean the worktree first" >&2; exit 1; }

# 1. Replace every occurrence of the secret across ALL history
#    (replace-text also rewrites the two ssh_*.py blobs of d3e84ca etc.)
printf '%s==>REMOVED<==\n' "$LEAKED_SECRET" > /tmp/leak_replacements.txt
git filter-repo --replace-text /tmp/leak_replacements.txt --force

# 2. Re-add the origin remote (filter-repo removes it on purpose) and
#    force-push ALL refs. Requires coordination with every collaborator.
git remote add origin https://github.com/yet4xs/compile-anything.git
git push --force --all origin
git push --force --tags origin

rm -f /tmp/leak_replacements.txt
echo "history rewritten and pushed. Verify with:"
echo "  git log --all -S'\$LEAKED_SECRET'   # must print nothing"
