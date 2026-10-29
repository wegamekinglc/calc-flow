#!/usr/bin/env sh
# Build-only compatibility for librdkafka's old minimum under CMake 4.
case "$1" in
  --build|--install|--version|-E|--help*) exec /usr/bin/cmake "$@" ;;
  *) exec /usr/bin/cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "$@" ;;
esac
