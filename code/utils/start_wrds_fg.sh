#!/bin/bash
# Foreground WRDS server — for the first interactive connect (Duo + any prompts).
# Once connected, LEAVE THIS WINDOW OPEN; the server runs here and the pipeline
# connects to it over the local socket (port 23847). Per-host, so one is enough.
cd "$(dirname "$0")/../.."
if [ -f .env ]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^[[:space:]]*# || -z "$key" ]] && continue
        value="${value%$'\r'}"
        export "$key=$value"
    done < .env
fi
export PGPASSWORD="$WRDS_PASS"
echo "Starting WRDS server in foreground."
echo "  - Approve the Duo push on your phone when it arrives."
echo "  - If it asks for a username, just press Enter (the default is correct)."
echo "  - Once it prints the connection is up, LEAVE THIS WINDOW OPEN."
echo
PYTHONPATH=code python3 code/utils/wrds_server.py
