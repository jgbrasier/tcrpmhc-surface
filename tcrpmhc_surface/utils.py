from typing import List, Tuple

import pandas as pd
import numpy as np

def hard_split_df(
        df: pd.DataFrame, target_col: str, min_ratio: float, random_seed: float, low: int, high: int, target_values: List[str]=None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """ Assume a target column, e.g. `epitope`.
    Then:
        1) Select random sample
        2) All samples sharing the same value of that column
        with the randomly selected sample are used for test
        3)Repeat until test budget (defined by train/test ratio) is
        filled.
    """
    if target_values:
        # if test target values are given, return train/test df directly
        train_df = df[~df[target_col].isin(target_values)]
        test_df = df[df[target_col].isin(target_values)]
        print("Train size = {:.2f}".format(len(train_df)/len(df)))
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True), target_values
    else:
        min_test_len = round((1-min_ratio) * len(df))
        test_len = 0
        selected_target_val = []

        train_df = df.copy()
        test_df = pd.DataFrame()
        
        target_count_df = df.groupby([target_col]).size().reset_index(name='counts')
        target_count_df = target_count_df[target_count_df['counts'].between(low, high, inclusive='both')]
        possible_target_val = list(target_count_df[target_col].unique())
        max_target_len = len(possible_target_val)

        while test_len < min_test_len:
    #         sample = train_df.sample(n=1, random_state=random_state)
    #         target_val = sample[target_col].values[0]
            rng = np.random.default_rng(seed=random_seed)
            target_val = rng.choice(possible_target_val)

            if target_val not in selected_target_val:
                to_test = train_df[train_df[target_col] == target_val]

                train_df = train_df.drop(to_test.index)
                test_df = pd.concat((test_df, to_test), axis=0)
                test_len = len(test_df)

                selected_target_val.append(target_val)
                possible_target_val.remove(target_val)

            if len(selected_target_val) == max_target_len:
                print(f"Possible targets left {possible_target_val}")
                raise Exception('No more values to sample from.')

        print(f"Target {target_col} sequences: {selected_target_val}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), selected_target_val

