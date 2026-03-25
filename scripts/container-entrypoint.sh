#!/bin/sh
# OpenMP / sklearn workaround: only on 64-bit ARM Linux (e.g. DGX Spark).
# x86_64 and other arches: leave LD_PRELOAD unset.
# Override: set LD_PRELOAD in the environment before start (docker compose / -e).
if [ -z "${LD_PRELOAD:-}" ] && [ "$(uname -m)" = "aarch64" ]; then
  _sys_gomp=/usr/lib/aarch64-linux-gnu/libgomp.so.1
  _sklearn_gomp=$(ls /usr/local/lib/python3.12/dist-packages/scikit_learn.libs/libgomp-*.so.* 2>/dev/null | head -1)
  if [ -f "$_sys_gomp" ] && [ -n "$_sklearn_gomp" ]; then
    export LD_PRELOAD="$_sys_gomp:$_sklearn_gomp"
  fi
fi
exec "$@"
