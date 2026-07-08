#!/usr/bin/env bash
# Commit local changes, push to the git remote that the elice server clones,
# then SSH into elice and pull so the remote checkout matches local.
#
# Usage:
#   ./sync_to_elice.sh "commit message"      # commit (if needed) + push + remote pull
#   ./sync_to_elice.sh                        # uses a default commit message
#
# Override any of these via environment variables:
#   GIT_REMOTE   local git remote to push to        (default: tda-target)
#   BRANCH       branch to push/pull                 (default: current branch)
#   SSH_KEY      path to the elice ssh key           (default: ~/.ssh/elice_friend.pem)
#   SSH_PORT     ssh port                            (default: 47537)
#   SSH_HOST     user@host                           (default: elicer@central-01.tcp.tunnel.elice.io)
#   REMOTE_DIR   git repo dir on elice               (default: ~/jonghyun/Topological-Data-Analysis)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GIT_REMOTE="${GIT_REMOTE:-tda-target}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/elice_friend.pem}"
SSH_PORT="${SSH_PORT:-47537}"
SSH_HOST="${SSH_HOST:-elicer@central-01.tcp.tunnel.elice.io}"
REMOTE_DIR="${REMOTE_DIR:-~/jonghyun/Topological-Data-Analysis}"
COMMIT_MSG="${1:-sync: update training/sweep scripts}"

echo ">> Repo:   $ROOT"
echo ">> Branch: $BRANCH -> $GIT_REMOTE"
echo ">> Elice:  $SSH_HOST:$REMOTE_DIR"

# 1) Commit local changes (only if the working tree is dirty).
if [[ -n "$(git status --porcelain)" ]]; then
  echo ">> Committing local changes..."
  git add -A
  git commit -m "$COMMIT_MSG"
else
  echo ">> Nothing to commit; working tree clean."
fi

# 2) Push to the remote that elice pulls from.
echo ">> Pushing to $GIT_REMOTE/$BRANCH..."
git push "$GIT_REMOTE" "$BRANCH"

# 3) On elice: fetch + checkout + pull (fast-forward) the same branch.
echo ">> Pulling on elice..."
ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_HOST" \
  "cd $REMOTE_DIR && git fetch --all --prune && git checkout $BRANCH && git pull --ff-only"

echo ">> Done. Remote checkout is now up to date on branch $BRANCH."
