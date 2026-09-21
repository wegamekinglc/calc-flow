#!/usr/bin/env bash
# Observe the pinned action's install command without replacing its setup logic.
if ! type -P rustup >/dev/null; then
    return 0
fi
rustup() {
    if [[ ${1:-} != toolchain || ${2:-} != install ]]; then
        command rustup "$@"
        return "$?"
    fi
    local evidence="${RUST_TOOLCHAIN_EVIDENCE:?}"
    local -a install_status
    mkdir -p "$evidence" || true
    { printf '%q ' rustup "$@"; printf '\n'; } > "$evidence/install-command.txt" || true
    if command rustup "$@" 2>&1 | tee --output-error=warn "$evidence/install.log"; then
        install_status=("${PIPESTATUS[@]}")
    else
        install_status=("${PIPESTATUS[@]}")
    fi
    printf '%s\n' "${install_status[0]}" > "$evidence/install-exit-code.txt" || true
    printf '%s\n' "${install_status[1]}" > "$evidence/log-exit-code.txt" || true
    return "${install_status[0]}"
}
