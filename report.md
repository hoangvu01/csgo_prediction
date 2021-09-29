## CS:GO Outcome Prediction Log & Report

### Third Draft
Column | Description
-------|------------
team_1_id | Unique identifier for team 1
team_2_id | Unique identifier for team 2
rank_1 | Team 1 HLTV Rank at the time of the match-up
rank_2 | Team 2 HLTV Rank  at the time of the match-up
best_of | Number of maps in the series `1/3/5`
map_id | Name of the map `de_dust2/de_mirage/etc...`
ct_team | Which team is currently on CT side
map_wins_1 | Number of wins on this map by team 1
map_wins_2 | Number of wins on this map by team 2
t1_score | Team 1 current score
t2_score | Team 2 current score
t1_equipment | Total value of equipment for team 1 before round starts
t2_equipment | Total value of equipment for team 2 before round starts
t1_streak | Number of round wins in a row before this round for team 1
t2_streak | Number of round wins in a row before this round for team 2
rating_p1_t1 | HLTV rating for player 1 of team 1 on this map
rating_p1_t2 | HLTV rating for player 1 of team 2 on this map
... | ...
rating_p5_t1 | HLTV rating for player 5 of team 1 on this map
rating_p5_t2 | HLTV rating for player 5 of team 2 on this map
**round_winner** | Which team has won the round - ultimate goal to predict

__Summary__:
- `t[1/2]_streak`: attempts to capture the momentum and morale and each team in the match
- `map_wins_[1/2]`: captures the experience and overall result of the team on the current map
- `rating_p_t`: captures the individual performances of team members on the current map

__Results__:
- The new features show a mixed result in performances across different models.
- The general trend is that performance worsens for `K-Neighbours, MLP Classifier and Naive Bayes`
- `Decision Tree` and `Random Forest` show slightly improvement change to performance.
- `AdaBoost` shows a consistenly high performance of **above 90%** accuracy across different runs and data partitions. More details are shown below.

#### AdaBoost **90% accuracy**

Visualisations here

--- 

### Second Draft
Column | Description
-------|------------
team_1_id | Unique identifier for team 1
team_2_id | Unique identifier for team 2
rank_1 | Team 1 HLTV Rank at the time of the match-up
rank_2 | Team 2 HLTV Rank  at the time of the match-up
best_of | Number of maps in the series `1/3/5`
map_id | Name of the map `de_dust2/de_mirage/etc...`
ct_team | Which team is currently on CT side
t1_score | Team 1 current score
t2_score | Team 2 current score
t1_equipment | Total value of equipment for team 1 before round starts
t2_equipment | Total value of equipment for team 2 before round starts
**round_winner** | Which team has won the round - ultimate goal to predict

__Summary__:
- Removed all the individual scores and equipment of previous rounds

__Results__:
- This simple set of data gives a much better performance as shown in the table below, with a peak performance of 62% accuracy.
- The dataset contains nearly 700k datapoints with `team_1` label occuring **49.9%** of the time and `team_2` **50.1%**. 

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