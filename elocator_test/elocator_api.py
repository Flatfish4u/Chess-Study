"""Elocator API integration functions"""
import requests
import logging
from time import sleep

def get_position_complexity_with_retry(fen, max_retries=3):
    """Get position complexity from Elocator API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://elocator.fly.dev/complexity/",
                json={"fen": fen}
            )
            if response.status_code == 200:
                sleep(0.1)  # 100ms delay between requests
                return response.json()["complexity_score"]
            logging.warning(f"API returned status code {response.status_code}")
        except Exception as e:
            logging.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
        sleep(1)  # Wait longer between retries
    return None

def analyze_game_complexity(pgn):
    """Get full game analysis from Elocator API"""
    try:
        response = requests.post(
            "https://elocator.fly.dev/analyze-game/",
            json={"pgn": pgn}
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        logging.warning(f"Failed to analyze game complexity: {e}")
        return None