def get_last_n_ratings_by_user(df, n):
    new_df = df.groupby('user').filter(lambda x: len(x) >= n).sort_values('timestamp').groupby('user').tail(n).sort_values('user') #use rank()
    return new_df
    
def mark_last_n_ratings_as_validation_set(
    df, n):
    df["is_valid"] = False
    df.loc[
        get_last_n_ratings_by_user(df, n).index,
        "is_valid",
    ] = True
    return df
