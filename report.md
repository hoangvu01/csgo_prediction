## CS:GO Outcome Prediction Log & Report

### First Draft

Column | Description
-------|------------
team_1_id | Unique identifier for team 1
team_2_id | Unique identifier for team 2
rank_1 | Team 1 HLTV Rank at the time of the match-up
rank_2 | Team 2 HLTV Rank  at the time of the match-up
best_of | Number of maps in the series `1/3/5`
map_id | Name of the map `de_dust2/de_mirage/etc...`
starting_ct | Which team starts on CTs side first
1_t1 | Total equipment value for team 1 in round 1
1_t2 | Total equipment value for team 2 in round 1
1_winner | Round winner (team 1 / team 2)
2_t1 | Total equipment value for team 1 in round 2
2_t2 | Total equipment value for team 2 in round 2
2_winner | Round winner (team 1 / team 2)
...|...
30_t1 | Total equipment value for team 1 in round 30
30_t2 |  Total equipment value for team 2 in round 30 
30_winner | Round winner (team 1 / team 2)
**round_winner** | Which team has won the round - ultimate goal to predict

__Explantion__:
- The few columns are self-explanatory
- Say the current round is `13`, all the columns from `1_t1, 1_t2, 1_winner`
to `12_t1, 12_t2, 12_winner` are all populated, showing all known results about the current match. 
`13_t1` and `13_t2` are also populated since before the round starts, we know how much each team invested whereas `13_winner` is blank since this is what we want to predict. 
All remaining columns up to `30_t1, 30_t2, 30_winner` are blank.

__Results__:
- After running this experiment with all available binary classifier on `sklearn`, the vast majority of them show completely random guesses, having accuracy of around 50%.
- This might be a result from the _curse of dimensionality_ since the data has over **90 features**.
- One of the reason for using `{n}_t1 {n}_t2 {n}_winner` format is to hopefully capture the momentum of each time as teams usually perform better when they are on a winning streak.  