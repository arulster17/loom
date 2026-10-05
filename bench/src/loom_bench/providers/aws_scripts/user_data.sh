# requires: TTL_EPOCH STAGE_FILE CLIENT_UID
# EC2 user-data. Anyone who can describe the instance can read this: no secrets.

mkdir -p "$(dirname "$STAGE_FILE")"
stage user_data_started

# Hard backstop: power off at the TTL even if the runner dies. The instance is
# launched with shutdown behaviour "terminate", so power-off ends billing.
remaining_min=$(( (TTL_EPOCH - $(date +%s)) / 60 ))
if [ "$remaining_min" -lt 1 ]; then
  shutdown -h now
  exit 0
fi
shutdown -h "+$remaining_min"

# The load-generator container uses host networking, so the IMDS hop limit does
# not stop it; deny its uid the metadata service (and with it the instance role).
iptables -I OUTPUT -d 169.254.169.254 -m owner --uid-owner "$CLIENT_UID" -j REJECT

stage user_data_done
