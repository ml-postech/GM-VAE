mkdir data/
wget https://www.dropbox.com/s/hyq44euztzz23o8/breakout_states_v2.h5
python tasks/Breakout/preprocessing.py
python tasks/CUB/preprocessing.py
python tasks/Oxford102/preprocessing.py
python tasks/Food101/preprocessing.py
