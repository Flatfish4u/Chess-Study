

I was wondering if we could categorize moves and lines by creativity or traps. I read a paper on stockfish identifying creativity but that was a load of bs as it tries to match human-identified brilliant moves with top engine line. Maybe we could work out a criteria for identification? 

PGN is not native for stockfish. It prefers pure algebraic notation. I could come up with an easy translation program for that but I'll probably go over the code once more sometime tomorrow or next week.


### 11/10/24

Currently breaking up that massive file into separate files to work on manageability 

There is a temporary function that starts to work on "creativity" and "traps," but I need to find some more papers and resources on it 


### 11/13/24

Okay so putting it in one jupyter file actually got it working. After code has ran in Jupyter, you can simply extract the data as CSV's; obviously saves time from not running code each time 

Next thing I'm looking at is optimal plots - after I know "how I want to frame the data," then I can go back into the actual data_processing function to begin twewaking numbers to get a more true mathematical representaiotn of what actually needs to happen

(1) How and by how-much do I need to expand the general population data-set to get more accurate numbers? 
    ie, is it better to (1) instead of random sampling games from the LiChess PGN, is it better to get players with similar amount of games, and use those to directly compare?
    this is a question that relies on statistical knowledge. I've been brushing up on t-tests, chi-squared, and hypothesis testing for this. This is not a hard answer, but rather a choice 

(2) Fixing the "traps and creativity" section from the data_processing.py 