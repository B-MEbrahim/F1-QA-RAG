import fastf1
import pandas as pd
from datetime import datetime

fastf1.Cache.enable_cache('data/cache')

def get_race_results(year: int, gp_name: str, top_k: int):
    """
    Fetches the top 10 race results for a specific Grand Prix.
    """
    try:
        # load session
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(telemetry=False, weather=True, messages=False)

        # get results 
        results = session.results
        
        # filter top 10
        top_k = results.head(top_k)[['Position', 'FullName', 'TeamName', 'Points']]

        # convert to markdow 
        return top_k.to_markdown(index=False)
    except Exception as e:
        return f"Error fetching results for {year} {gp_name}: {str(e)}"
    

if __name__ == "__main__":
    print("--- Testing Bahrain 2024 ---")
    res = get_race_results(2025, 'Bahrain', 3)
    print(res)