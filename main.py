from src.train import train_all
from src.evaluate import evaluate_all
from src.backtest import backtest_all

def main():
  train_all()
  evaluate_all()
  backtest_all()

if __name__ == "__main__":
  main()
