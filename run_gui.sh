set -e

python planet-wars-rts/app/src/main/python/client_server/game_agent_server.py &
sleep 5

cd planet-wars-rts
./gradlew :app:runGUI
