from src.prepare_data import prepare
from src.eda import run_eda
from src.decision_trees import run_trees
from src.association_rules import run_association
from src.regression import run_regression

def main():
    print("\n[1/5] Preparing data")
    prepare()
    print("\n[2/5] Exploratory analysis")
    run_eda()
    print("\n[3/5] Decision-tree comparison")
    run_trees()
    print("\n[4/5] Association-rule mining")
    run_association()
    print("\n[5/5] Correlation and regression")
    run_regression()
    print("\nAll tasks completed. Check the outputs/ and models/ folders.")

if __name__ == "__main__":
    main()
