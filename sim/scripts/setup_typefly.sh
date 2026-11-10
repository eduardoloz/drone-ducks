#!/usr/bin/env bash
# One-time (re-runnable): TypeFly at a pinned upstream commit + our PX4 patch, in sim/.deps/TypeFly,
# installed into sim/.venv together with torch / ultralytics (~1-2 GB download the first time).
#
# Why a pinned checkout + patch instead of a fork: we have no push access to typefly/TypeFly, and
# the patch is small (sim/typefly/patches/). All PX4 flight code lives in quackdrone; TypeFly only
# gets a 3-line shim. Upgrading TypeFly = bump TYPEFLY_COMMIT and make the patch apply again.
set -euo pipefail
source "$(dirname "$0")/common.sh"
need_venv

TYPEFLY_REPO="${TYPEFLY_REPO:-https://github.com/typefly/TypeFly.git}"
TYPEFLY_COMMIT="${TYPEFLY_COMMIT:-ef1f5cc2555c8d2d2f9035516152baab66b924c4}"   # upstream main, 2026-11-10
PATCH="$SIM_DIR/typefly/patches/0001-quack-px4-platform.patch"

echo "== TypeFly checkout ($TYPEFLY_DIR @ ${TYPEFLY_COMMIT:0:7}) =="
if [ ! -d "$TYPEFLY_DIR/.git" ]; then
  mkdir -p "$(dirname "$TYPEFLY_DIR")"
  git clone "$TYPEFLY_REPO" "$TYPEFLY_DIR"
fi
cd "$TYPEFLY_DIR"
if git apply --reverse --check "$PATCH" 2>/dev/null && [ "$(git rev-parse HEAD)" = "$TYPEFLY_COMMIT" ]; then
  echo "patch already applied"
else
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    [ -n "${FORCE:-}" ] || die "$TYPEFLY_DIR has local changes. Commit/stash them, or FORCE=1 to discard."
    git reset -q --hard
  fi
  git cat-file -e "$TYPEFLY_COMMIT^{commit}" 2>/dev/null || git fetch -q origin
  git checkout -q --detach "$TYPEFLY_COMMIT"
  git clean -fdq typefly/platforms typefly/config
  git apply "$PATCH"
  echo "applied $(basename "$PATCH")"
fi

echo "== install into $SIM_DIR/.venv =="
# Intel macOS: torch 2.2.2 is the last x86_64 mac wheel and needs NumPy 1.x (and so OpenCV 4.11).
PINS=("numpy<2")
if [ "$(uname)" = "Darwin" ] && [ "$(uname -m)" = "x86_64" ]; then
  PINS+=("torch==2.2.2" "torchvision==0.17.2" "opencv-python<4.12")
fi
uv pip install --python "$SIM_PYTHON" -e "$TYPEFLY_DIR[dev]" -e "$SIM_DIR[dev,vision]" "${PINS[@]}"

# Reuse already-downloaded YOLO weights if Eddie's setup has them (saves a download).
for w in "$HOME/ProjectQuack/setup/TypeFly/typefly/serving/models/"*.pt; do
  [ -f "$w" ] && [ ! -f "$TYPEFLY_DIR/typefly/serving/models/$(basename "$w")" ] && \
    cp "$w" "$TYPEFLY_DIR/typefly/serving/models/" && echo "copied $(basename "$w")"
done
[ -f "$TYPEFLY_DIR/.env" ] || { cp "$TYPEFLY_DIR/.env.example" "$TYPEFLY_DIR/.env"; echo "created $TYPEFLY_DIR/.env (put OPENAI_API_KEY there for the web UI)"; }

echo "== TypeFly unit tests =="
(cd "$TYPEFLY_DIR" && "$SIM_PYTHON" -m pytest -q 2>&1 | tail -3) || echo "WARNING: TypeFly tests failed (see above)"
cat <<MSG

TypeFly + PX4 ready.
  sim/scripts/start_all.sh                 # sim + camera + YOLO + web UI (http://127.0.0.1:50000, needs OPENAI_API_KEY)
  NO_WEBUI=1 sim/scripts/start_all.sh      # everything except the LLM web UI (no API cost)
  sim/scripts/py.sh sim/typefly/e2e_check.py --no-llm     # "find the chair and fly to it", canned plan
MSG
