from src.tournament_engine_v3_1 import V31RealTimeTournamentEngine
import json

engine = V31RealTimeTournamentEngine(price_area="DK1", initial_capital=100000, profile="tier3_aggressive")
res = engine.run_tournament()
print(res.keys())
if 'leaderboard' in res and len(res['leaderboard']) > 0:
    # Print the keys for the first model in the leaderboard to see if we can get quarter predictions
    print("Leaderboard keys:", res['leaderboard'][0].keys())
    if 'ledger' in res['leaderboard'][0]:
        print("Ledger example:", res['leaderboard'][0]['ledger'][0])
