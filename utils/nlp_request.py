from utils.df_async_post import DfAsyncPost
from utils import nlp
import pandas as pd


class Nlp:
    def __init__(self):
        pass

    @property
    def seg(self):
        return DfAsyncPost(
            url="",
            payload="{\"query\":\"%s\"}",
            df_response="nlp_seg"
        )


def init_df():
    df = pd.read_excel()
    return df


if __name__ == '__main__':
    df = init_df()
    df = nlp.run(df, "用户问")
    print(df[:5])
