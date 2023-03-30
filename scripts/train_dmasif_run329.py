
import pandas as pd


PDB_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/pdb/run329_results_for_jg"
PROCESSED_DIR = "/n/data1/hms/dbmi/zitnik/lab/users/jb611/surface/run_329"
TSV_PATH = "data/preprocessed/run329_results.tsv"

data = pd.read_csv(TSV_PATH, sep='\t')
# select only positive samples
data = data[data['binder']==1].copy()
