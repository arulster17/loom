# requires: CONTAINER
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
echo "loom-stopped $CONTAINER"
