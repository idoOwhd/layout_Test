#!/usr/bin/env bash
# Turn layout_research itself into a Git working tree and publish source only.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/liangyilei/ladder_home/staged/baseline_framework/layout_research}"
REMOTE_URL="${REMOTE_URL:-git@github.com:idoOwhd/layout_Test.git}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
GIT_AUTHOR_NAME_LOCAL="${GIT_AUTHOR_NAME_LOCAL:-idoOwhd}"
GIT_AUTHOR_EMAIL_LOCAL="${GIT_AUTHOR_EMAIL_LOCAL:-idoOwhd@users.noreply.github.com}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Publish layout research implementation}"

[[ -d "${REPO_DIR}" ]] || { echo "repository directory missing: ${REPO_DIR}" >&2; exit 2; }
command -v git >/dev/null || { echo "git is required" >&2; exit 2; }
cd "${REPO_DIR}"

if [[ ! -d .git ]]; then
  echo "[publish] initializing Git working tree in ${REPO_DIR}"
  git init -b "${TARGET_BRANCH}"
  git remote add origin "${REMOTE_URL}"
  git fetch origin "${TARGET_BRANCH}"
  if git show-ref --verify --quiet "refs/remotes/origin/${TARGET_BRANCH}"; then
    # Adopt remote history without changing any current source/result file.
    git update-ref "refs/heads/${TARGET_BRANCH}" "refs/remotes/origin/${TARGET_BRANCH}"
    git reset --mixed "refs/remotes/origin/${TARGET_BRANCH}"
  fi
else
  EXISTING_REMOTE="$(git remote get-url origin 2>/dev/null || true)"
  if [[ -z "${EXISTING_REMOTE}" ]]; then
    git remote add origin "${REMOTE_URL}"
  elif [[ "${EXISTING_REMOTE}" != "${REMOTE_URL}" ]]; then
    echo "origin is ${EXISTING_REMOTE}, expected ${REMOTE_URL}; refusing to replace it" >&2
    exit 2
  fi
  git fetch origin "${TARGET_BRANCH}"
fi

git switch "${TARGET_BRANCH}" >/dev/null 2>&1 || git switch -c "${TARGET_BRANCH}"
git branch --set-upstream-to="origin/${TARGET_BRANCH}" "${TARGET_BRANCH}" \
  >/dev/null 2>&1 || true

if git show-ref --verify --quiet "refs/remotes/origin/${TARGET_BRANCH}"; then
  if ! git merge-base --is-ancestor "origin/${TARGET_BRANCH}" HEAD; then
    echo "local and remote histories diverged (or remote is ahead); reconcile before publishing" >&2
    exit 2
  fi
fi

git config user.name "${GIT_AUTHOR_NAME_LOCAL}"
git config user.email "${GIT_AUTHOR_EMAIL_LOCAL}"
git add -A

FORBIDDEN_RE='(^|/)(results|models|framework_envs|framework_sources|framework_cache|framework_install_logs)(/|$)|(^|/)framework_envs\.generated\.sh$|(^|/)__pycache__(/|$)|\.pyc$'
if git diff --cached --name-only | grep -Eq "${FORBIDDEN_RE}"; then
  echo "refusing to commit generated results, models, environments, clones, or caches:" >&2
  git diff --cached --name-only | grep -E "${FORBIDDEN_RE}" >&2
  exit 3
fi

AUDIT_DIR="$(mktemp -d /tmp/layout-test-audit.XXXXXX)"
cleanup() {
  case "${AUDIT_DIR}" in
    /tmp/layout-test-audit.*) rm -rf -- "${AUDIT_DIR}" ;;
    *) echo "refusing to remove unexpected audit path: ${AUDIT_DIR}" >&2 ;;
  esac
}
trap cleanup EXIT

LARGE_FILE_LIST="${AUDIT_DIR}/large_files.txt"
find "${REPO_DIR}" -path "${REPO_DIR}/.git" -prune -o \
  -path "${REPO_DIR}/results" -prune -o \
  -path "${REPO_DIR}/models" -prune -o \
  -path "${REPO_DIR}/framework_envs" -prune -o \
  -path "${REPO_DIR}/framework_sources" -prune -o \
  -path "${REPO_DIR}/framework_cache" -prune -o \
  -type f -size +25M -print >"${LARGE_FILE_LIST}"
if [[ -s "${LARGE_FILE_LIST}" ]]; then
  echo "refusing to commit source files larger than 25 MiB:" >&2
  sed "s#^${REPO_DIR}/##" "${LARGE_FILE_LIST}" >&2
  exit 3
fi

SECRET_HITS="${AUDIT_DIR}/secret_hits.txt"
set +e
grep -RIlE \
  --exclude-dir=.git --exclude-dir=results --exclude-dir=models \
  --exclude-dir=framework_envs --exclude-dir=framework_sources \
  --exclude-dir=framework_cache \
  --exclude='push_layout_research_to_github.sh' \
  '(github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----)' \
  "${REPO_DIR}" >"${SECRET_HITS}"
SECRET_SCAN_CODE=$?
set -e
if [[ "${SECRET_SCAN_CODE}" == 0 ]]; then
  echo "possible credential material found; refusing to commit:" >&2
  sed "s#^${REPO_DIR}/##" "${SECRET_HITS}" >&2
  exit 3
elif [[ "${SECRET_SCAN_CODE}" != 1 ]]; then
  echo "credential scan failed" >&2
  exit 3
fi

if git diff --cached --quiet; then
  echo "[publish] no source changes to commit"
else
  echo "[publish] staged files: $(git diff --cached --name-only | wc -l)"
  git diff --cached --stat
  git commit -m "${COMMIT_MESSAGE}"
fi

echo "[publish] pushing ${TARGET_BRANCH}"
git push origin "HEAD:${TARGET_BRANCH}"
echo "[publish] commit=$(git rev-parse HEAD)"
echo "[publish] remote=$(git remote get-url origin) branch=$(git branch --show-current)"
