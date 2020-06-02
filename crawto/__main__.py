import argparse
import os

import pandas as pd
import papermill as pm

from crawto.data_cleaning_flow import (data_cleaning_flow,
                                       run_data_cleaning_flow)
from crawto.meta_model import meta_model_flow, run_meta_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("problem")
    parser.add_argument("target")
    parser.add_argument("-d", "--db_name", default="crawto.db")
    args = parser.parse_args()
    csv = args.csv
    problem = args.problem
    target = args.target
    db_name = args.db_name
    if os.path.isfile(db_name):
        response = input(
            f'This will override your current {args.db_name}.\n Continue? y/n:\n'
        )

        if response == 'y':
            os.remove(db_name)
            flow(args)
        elif response != 'n':
            print("unable to parse input. please enter 'y' or 'n'.")
    else:
        flow(args)


def flow(args):
    df = pd.read_csv(args.csv)
    run_data_cleaning_flow(
        data_cleaning_flow,
        input_df=df,
        problem=args.problem,
        target=args.target,
        db_name=args.db_name,
    )
    run_meta_model(meta_model_flow, problem=args.problem, db_name=args.db_name)


if __name__ == "__main__":

    main()
