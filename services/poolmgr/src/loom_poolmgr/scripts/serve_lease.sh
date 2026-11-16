# requires: LEASE_FILE LEASE_EPOCH
# Renew this pod's lease: the watchdog in serve_start terminates the pod once
# now >= max(LOOM_LEASE_EPOCH, contents of LEASE_FILE). Runs as root over SSH.
#
# The write is atomic: a temporary file in the lease file's own directory (never under
# $TMPDIR, which may be another filesystem) renamed over it, so the watchdog reads the
# old epoch or the new one, never a partial or missing file. On a fresh pod the
# directory (/run/loom) does not exist yet. Any failure exits non-zero: the manager
# retries the renewal on its next tick.

[[ "$LEASE_EPOCH" =~ ^[0-9]{1,18}$ ]] || fail "LEASE_EPOCH is not an epoch: $LEASE_EPOCH"
lease_dir="$(dirname -- "$LEASE_FILE")"
mkdir -p -- "$lease_dir"
tmp="$(mktemp -- "$lease_dir/.lease.XXXXXX")"
trap 'rm -f -- "$tmp"' EXIT
printf '%s\n' "$LEASE_EPOCH" >"$tmp"
chmod 644 -- "$tmp"
mv -f -- "$tmp" "$LEASE_FILE"
trap - EXIT
echo "loom-lease $LEASE_EPOCH"
