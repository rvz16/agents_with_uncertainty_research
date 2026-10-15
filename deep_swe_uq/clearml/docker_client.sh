#!/usr/bin/env bash
# Put a docker client on PATH, or fail loudly. Sourced, not executed.
#
# Kept in one place because getting it wrong is expensive and silent: the first
# version tried apt twice and get.docker.com once, each with output to /dev/null
# and each ending in "|| true", so four DG shards died reporting only
# "docker: command not found".
# Only the client is needed: the daemon is the host's, reached through the mounted
# socket. apt installed it on the earlier runs and then silently failed on this
# worker, and because every attempt was "|| true" with output to /dev/null the log
# said nothing but "command not found". So: report each attempt, and prefer the
# static binary, which needs no package manager and no mirror.
DOCKER_CLI_VERSION="${DOCKER_CLI_VERSION:-29.9.0}"
if ! command -v docker >/dev/null 2>&1; then
  echo "[grade] no docker client, installing"
  arch="$(uname -m)"
  url="https://download.docker.com/linux/static/stable/${arch}/docker-${DOCKER_CLI_VERSION}.tgz"
  if curl -fsSL --max-time 300 "$url" 2>/tmp/dl.err \
       | tar xz -C /tmp --strip-components=1 docker/docker 2>/tmp/tar.err \
       && install -m 0755 /tmp/docker /usr/local/bin/docker; then
    echo "[grade] installed the static client ${DOCKER_CLI_VERSION} for ${arch}"
  else
    echo "[grade] static client failed: $(tail -c 200 /tmp/dl.err 2>/dev/null) $(tail -c 200 /tmp/tar.err 2>/dev/null)"
    apt-get update -qq -o Acquire::AllowInsecureRepositories=true >/tmp/apt.log 2>&1 \
      || echo "[grade] apt-get update failed: $(tail -c 200 /tmp/apt.log)"
    apt-get install -y -qq --no-install-recommends docker.io >>/tmp/apt.log 2>&1 \
      || echo "[grade] apt-get install failed: $(tail -c 300 /tmp/apt.log)"
  fi
  command -v docker >/dev/null 2>&1 \
    || curl -fsSL --max-time 300 https://get.docker.com 2>/tmp/get.err | sh >/tmp/get.log 2>&1 \
    || echo "[grade] get.docker.com failed: $(tail -c 200 /tmp/get.err 2>/dev/null) $(tail -c 200 /tmp/get.log 2>/dev/null)"
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[grade] VERDICT: no docker client could be installed on this worker"; exit 21
fi

ls -la /var/run/docker.sock 2>&1 | head -2
docker version --format 'client={{.Client.Version}} server={{.Server.Version}}' 2>&1 | head -2
if ! docker ps >/dev/null 2>&1; then
  echo "[docker] VERDICT: daemon unreachable from this worker"; return 20 2>/dev/null || exit 20
fi
echo "[docker] reachable, arch $(uname -m)"
