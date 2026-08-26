from src.train import train_all
from src.evaluate import evaluate_all
from src.backtest import backtest_all

def main():
  train_all(suffix='_features', feature_set='extended')
  evaluate_all(suffix='_features', feature_set='extended')
  backtest_all(suffix='_features', feature_set='extended')

if __name__ == "__main__":
  main()
