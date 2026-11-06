#!/usr/bin/env bash
# Save one bot's result straight away, so a run that times out keeps the bots it finished. Starts
# from the latest main each try, so a merge during a long run cannot lose results.
set -euo pipefail

bot=$1
file="results/$bot.json"
[ -n "$(git status --porcelain -- "$file")" ] || exit 0

keep=$(mktemp)
cp "$file" "$keep"
for attempt in 1 2 3 4 5; do
  git fetch -q origin main
  git reset -q --hard origin/main
  cp "$keep" "$file"
  uv run python -m arena.leaderboard >/dev/null
  git add results site
  git commit -qm "ladder: $bot"
  git push -q origin HEAD:main && exit 0
  sleep $((attempt * 15))
done
echo "could not push the result for $bot" >&2
exit 1
