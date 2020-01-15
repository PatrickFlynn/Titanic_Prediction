@echo off
set /p id="Comment: "
S: && cd "S:\Code\Anaconda\Scripts" && activate DataScience && cd "S:\Code\Data Science\Titanic_Prediction" && kaggle competitions submit -c titanic -f submission.csv -m "%id%" && TIMEOUT 120 && kaggle competitions submissions -c titanic > Kaggle_Submits.txt