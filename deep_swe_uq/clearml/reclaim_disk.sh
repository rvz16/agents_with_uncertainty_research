#!/usr/bin/env bash
# Report the worker's disk, then reclaim only what this project left behind.
#
# The daemon is shared with other teams. "docker system prune -a" would delete
# their images and can break a running job of theirs, so nothing here touches
# anything that is not ours: the verifier images we build are tagged
# deepswe-verif:*, and the replay containers are named replay_*. The build cache
# is the one shared thing we drop, because it is a cache -- the cost to someone
# else is a slower build, not a lost artifact.
#
# The snapshots under SHARED are deliberately kept: patches_* is replay output
# that cost hours and is what the grading step reads.
set -uo pipefail
SHARED="${SHARED:-/tmp/grade_shared}"
DO_CLEAN="${DO_CLEAN:-1}"

banner() { echo; echo "===== $* ====="; }

source "$(dirname "${BASH_SOURCE[0]}")/docker_client.sh" || exit 20

banner "disk before"
df -h / /tmp 2>/dev/null | sed 's/^/  /'
banner "docker storage before"
docker system df 2>&1 | sed 's/^/  /'

banner "what is ours"
ours_img=$(docker images --filter "reference=deepswe-verif:*" --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | sort)
echo "  verifier images: $(echo "$ours_img" | grep -c . 2>/dev/null)"
docker images --filter "reference=deepswe-verif:*" --format '    {{.Repository}}:{{.Tag}}  {{.Size}}' 2>/dev/null | head -40
ours_run=$(docker ps --filter "name=replay_" --format '{{.Names}}' 2>/dev/null)
ours_all=$(docker ps -a --filter "name=replay_" --format '{{.Names}}' 2>/dev/null)
echo "  replay containers: $(echo "$ours_all" | grep -c . 2>/dev/null) total, $(echo "$ours_run" | grep -c . 2>/dev/null) still running"
banner "snapshots we must keep"
du -sh "${SHARED}"/patches_* 2>/dev/null | sed 's/^/  /' || echo "  none"
du -sh "${SHARED}" 2>/dev/null | sed 's/^/  /'

if [ "$DO_CLEAN" != "1" ]; then echo; echo "[reclaim] DO_CLEAN=0, reporting only"; exit 0; fi

banner "removing our containers"
if [ -n "$ours_run" ]; then
  echo "  leaving these alone, they are running: $ours_run"
fi
for name in $(docker ps -a --filter "name=replay_" --filter "status=exited" --format '{{.Names}}' 2>/dev/null) \
            $(docker ps -a --filter "name=replay_" --filter "status=created" --format '{{.Names}}' 2>/dev/null); do
  docker rm -f "$name" >/dev/null 2>&1 && echo "  removed $name"
done

banner "removing our verifier images"
for image in $ours_img; do
  docker rmi -f "$image" >/dev/null 2>&1 && echo "  removed $image"
done

banner "dropping the build cache"
docker builder prune -f 2>&1 | tail -3 | sed 's/^/  /'

banner "disk after"
df -h / /tmp 2>/dev/null | sed 's/^/  /'
banner "docker storage after"
docker system df 2>&1 | sed 's/^/  /'
banner "snapshots still present"
du -sh "${SHARED}"/patches_* 2>/dev/null | sed 's/^/  /' || echo "  none"
echo; echo "[reclaim] done"
