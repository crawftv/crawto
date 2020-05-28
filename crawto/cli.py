import argparse
from meta_model import run_meta_model, meta_model_flow
from ml_flow import run_ml_flow, data_cleaning_flow
import pandas as pd

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
df = pd.read_csv(args.csv)
run_ml_flow(
    data_cleaning_flow,
    input_df=df,
    problem=args.problem,
    target=target,
    db_name=db_name,
)
run_meta_model(meta_model_flow, db_name=db_name)
