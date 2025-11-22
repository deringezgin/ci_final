set -euo pipefail

MODE="${1:-}"
echo "Adding the Python bindings to PYTHONPATH" 
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/planet-wars-rts/app/src/main/python"

echo "Killing the existing server"
pkill -f "client_server/game_agent_server.py" 2>/dev/null || true

echo "Starting the new server"
python3 planet-wars-rts/app/src/main/python/client_server/game_agent_server.py &
sleep 5

cd planet-wars-rts
if [ "$MODE" = "headless" ]; then
  xvfb-run ./gradlew :app:runGUI
else
  ./gradlew :app:runGUI
fi
