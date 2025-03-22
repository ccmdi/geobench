GeoBench is a benchmark for evaluating how well large language models can identify geographic locations from images through the context of GeoGuessr. This project tests whether models can generalize beyond their primary training modalities to perform spatial reasoning tasks.

**[Leaderboard](https://geobench.org)**

# Installation
```
git clone https://github.com/ccmdi/geobench.git
cd geobench
pip install -r requirements.txt
```

## Create a dataset
```
python dataset.py --num <n> --output <test name> --map <geoguessr map id>
```

## Test a model
```
python geobench.py --dataset <test name> --model <model name>
```

Models go by their class name in `models.py`. Claude 3.5 Haiku goes by `Claude3_5Haiku`, for instance.