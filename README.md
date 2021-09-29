## CS:GO Round Outcome Prediction

This is [hoangvu01's](https://github.com/hoangvu01) fun summer project aiming to:
- Gather data from [HLTV](https://www.hltv.org/)
- Clean, aggregate data and define new metrics
- Predict the winner of each round based on each team's economy and total equipment values at the beginning of the round

### 🎮 The Game
`Counter Strike: Global Offensive` is a _Tactical First Person Shooter (Tactical FPS)_ video game
where 2 opposing forces, the _Terroists (Ts)_ and _Counter-Terroists (CTs)_, go against each other.

The main objective for Ts side is to:
- Eliminate all CTs or...
- Successfully plant and detonate a bomb

while the CTs will try to eliminate the Ts and to prevent the bomb detonation.

Each game generally consists of 2 halves of 15 rounds where the CTs and Ts switch side
at each half. First team to 16 round wins the game. 

At the end of each round, each team get a certain amount of cash, determined by
how many round wins/losses they have had. This money can then used to purchase
weapons, grenades, armour and more... which we often call `team economy`. 

### 💾 Data
There are 2 sources of data for this project, the first set of data had been collected and 
[published on Kaggle](https://www.kaggle.com/austinpack/cs-go-esports-analysis/data) by 
[Austin Pack](https://www.kaggle.com/austinpack) for public usage. 

However, I have also written my own Python Module which is an unofficial HLTV API, 
used for collecting historical matches data including overall scores, map picks, 
team economies and more. You can check out the project [here](https://github.com/hoangvu01/hltv_python).

### 🧫 Experiment

- Data is first gathered using the `Kaggle Dataset` as described above
- Then transform the `CSV` files into useful data using the scripts in [`scripts/preprocess.py`](scripts/preprocess.py)
- Run `model.train.train_multi_model()`, saving the results and document findings.
- All findings and progress are documented in [`report.md`](report.md)