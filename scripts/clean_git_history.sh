#!/usr/bin/env bash
# DRAFT — history cleanup for the leaked SSH password. DO NOT RUN until
# every gate below is satisfied. Kept non-executable on purpose
# (git keeps the +x bit off; do not chmod +x it casually).
#
# Hard gates (enforced programmatically, not just in comments):
#   1. CREDENTIAL_ROTATED=yes   — the server owner confirms the leaked
#      credential was rotated at the provider console.
#   2. FORCE_PUSH_AUTHORIZED=yes — the owner explicitly authorizes the
#      public-history rewrite (all clones diverge afterwards).
#   3. Only 'main' is rewritten/pushed — never --all/--tags.
#   4. A verifiable backup bundle is created BEFORE the rewrite and its
#      path printed; if the rewrite fails, restore from the bundle.
#
# The secret is read from LEAKED_SECRET and written only to a mktemp file
# with 0600 perms, removed by an EXIT trap — never a fixed /tmp path.
set -euo pipefail

[ "${CREDENTIAL_ROTATED:-}" = "yes" ] || { echo "gate: CREDENTIAL_ROTATED != yes (rotate the credential at the provider console first)" >&2; exit 1; }
[ "${FORCE_PUSH_AUTHORIZED:-}" = "yes" ] || { echo "gate: FORCE_PUSH_AUTHORIZED != yes" >&2; exit 1; }
: "${LEAKED_SECRET:?set LEAKED_SECRET to the old (rotated) password}"
command -v git-filter-repo >/dev/null 2>&1 || { echo "pip install git-filter-repo" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "clean the worktree first" >&2; exit 1; }

BACKUP="../compile-anything-pre-rewrite-$(date +%Y%m%d%H%M%S).bundle"
git bundle create "$BACKUP" --all
echo "backup bundle: $BACKUP  (verify: git bundle verify $BACKUP)"

REPL="$(mktemp)"              # private 0600 temp file, not a fixed path
trap 'rm -f "$REPL"' EXIT
chmod 600 "$REPL"
printf '%s==>REMOVED<==\n' "$LEAKED_SECRET" > "$REPL"

git filter-repo --replace-text "$REPL" --force \
    --refs refs/heads/main    # main only

# Re-add origin (filter-repo removes it) and push MAIN ONLY.
git remote add origin https://github.com/yet4xs/compile-anything.git
git push --force origin main

echo "history rewritten. verify with (must print nothing):"
echo "  git log --all -S\"\$LEAKED_SECRET\""
echo "if anything is wrong, restore from the bundle:"
echo "  git clone $BACKUP recovered && cd recovered && git push --force origin main"
