# Updated ADHD usernames based on your research
ADHD_USERNAMES = [
    # Original list
    "teoeo",
    "Tobermorey", 
    "apostatlet",
    "LovePump1000",
    "StuntmanAndy",
    "ChessyChesterton12",
    "yastoon",
    "SonnyDayz11",
    "Xiroir",
    "StellaAthena",
    "MagikPigeon",
    
    # Additional from your spreadsheet (verified ADHD with sufficient games)
    "ctmadrock",  # Timy1976 equivalent
    "pawnsgoback",
    "spaceghostshivs", # Yastoon
    "Dru403",
    "ellehooq",
    "Euph4life", 
    "Matthew-Marchand",
    "Rosey12",
    "s0mething213",
    "B1SH0P_B1SH0P",
    "Wildwood",
    "Kanaan92",
    "jonesmh",
    
    # Add more as you verify their status and game counts
    # Priority: players with >1000 games and verified ADHD diagnosis
]

# Validation function to check if usernames exist and have sufficient games
def validate_usernames(usernames, min_games=300):
    """
    Validate that usernames exist and have sufficient games before processing
    """
    validated_users = []
    failed_users = []
    
    for username in usernames:
        try:
            # Quick API call to check if user exists
            url = f"https://lichess.org/api/user/{username}"
            response = requests.get(url)
            
            if response.status_code == 200:
                user_data = response.json()
                total_games = user_data.get('count', {}).get('all', 0)
                
                if total_games >= min_games:
                    validated_users.append(username)
                    print(f"✓ {username}: {total_games} games")
                else:
                    print(f"⚠ {username}: Only {total_games} games (below {min_games})")
            else:
                failed_users.append(username)
                print(f"✗ {username}: User not found (404)")
                
        except Exception as e:
            failed_users.append(username)
            print(f"✗ {username}: Error - {str(e)}")
    
    print(f"\nValidated: {len(validated_users)} users")
    print(f"Failed: {len(failed_users)} users")
    
    return validated_users, failed_users
