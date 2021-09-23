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

Data is first transformed into the following format:

Column | Description
-------|------------
team_1_id | Unique identifier for team 1
team_2_id | Unique identifier for team 2
rank_1 | Team 1 HLTV Rank at the time of the match-up
rank_2 | Team 2 HLTV Rank  at the time of the match-up
best_of | Number of maps in the series `1/3/5`
map_id | Name of the map `de_dust2/de_mirage/etc...`
starting_ct | Which team starts on CTs side first
t1_score | Number of rounds team 1 has won
t2_score | Number of rounds team 2 has won
t1_equipment | Total value of equipment for team 1 before round starts
t2_equipment | Total value of equipment for team 2 before round starts
**round_winner** | Which team has won the round - ultimate goal to predict



Below is the result of running `model.train.train_multi_model()`:

Model | Details | Training / Test Set Ratio | Accuracy with Training Data | Accuracy with Test Data 
------|---------|---------------------------|-----------------------------|------------------------
Nearest Neighbours | `KNeighborsClassifier(3)` | 70:30 | 0.7448211787769501 | 0.579183028044736 
Nearest Neighbours | `KNeighborsClassifier(3)` | 50:50 | 0.7442934458706685 | 0.5812889345224178
Nearest Neighbours | `KNeighborsClassifier(3)` | 30:70 | 0.7462574508107147| 0.5798076875788064
Linear SVM | `SVC(kernel="linear", C=0.025)` | 70:30 | 0.6124187401495538| 0.6146528110701572
Linear SVM | `SVC(kernel="linear", C=0.025)` | 50:50 | 0.6138215590456774| 0.6118148847204568
Linear SVM | `SVC(kernel="linear", C=0.025)` | 30:70 | 0.6111840833328729| 0.614272519685273572
RBF SVM | `SVC(gamma=2, C=1)` | 70:30 | 1.0| 0.5007920711593294
RBF SVM | `SVC(gamma=2, C=1)` | 50:50 | 1.0| 0.4981235073353804
RBF SVM | `SVC(gamma=2, C=1)` | 30:70 | 1.0| 0.5018479730550511
Decision Tree | `DecisionTreeClassifier(max_depth=5)` | 70:30 | 0.616298033326722| 0.6198542940408331
Decision Tree | `DecisionTreeClassifier(max_depth=5)` | 50:50 | 0.6073441154986365| 0.6022033293194575
Decision Tree | `DecisionTreeClassifier(max_depth=5)` | 30:70 | 0.6182158760641473| 0.6146452840666958
Random Forest | `RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)` | 70:30 | 0.6070526882938014| 0.5989344322013435
Random Forest | `RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)` | 50:50 | 0.6016906850015442| 0.5969586394288369
Random Forest | `RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)` | 30:70 | 0.6087195077904066| 0.5987261796207715
Neural Net | `MLPClassifier(alpha=1, max_iter=1000)` | 70:30 | 0.5951849578222274| 0.5961346910186581
Neural Net | `MLPClassifier(alpha=1, max_iter=1000)` | 50:50 | 0.6192380044563555| 0.6196964613961038
Neural Net | `MLPClassifier(alpha=1, max_iter=1000)` | 30:70 | 0.515330894175279 | 0.5146073359897259
AdaBoost | `AdaBoostClassifier()` | 70:30 | 0.6119837046189645| 0.6141008882015485
AdaBoost | `AdaBoostClassifier()` | 50:50 | 0.6192619063155188| 0.6084604924337447
AdaBoost | `AdaBoostClassifier()` | 30:70 | 0.6189462043236207| 0.610070374044156
NaiveBayes | `GaussianNB()` | 70:30 | 0.6072810156305355| 0.6131758643754753
NaiveBayes | `GaussianNB()` | 50:50 | 0.608811215508755| 0.6017055350344511
NaiveBayes | `GaussianNB()` | 30:70 | 0.608323728486449| 0.6062629024591133
QDA | `QuadraticDiscriminantAnalysis()` | 70:30 | 0.6103008089695579| 0.6084014281883848
QDA | `QuadraticDiscriminantAnalysis()` | 50:50 | 0.6106047741463634| 0.6100988616049322
QDA | `QuadraticDiscriminantAnalysis()` | 30:70 | 0.6092474406112338| 0.6091018645017066


