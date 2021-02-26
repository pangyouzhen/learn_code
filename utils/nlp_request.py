from utils.df_async_post import DfAsyncPost
from utils import nlp
import pandas as pd

url = ""
payload = "{\"query\":\"%s\"}"


class Nlp(DfAsyncPost):
    def __init__(self, url, payload, df_response):
        super().__init__(url, payload, df_response)

    @classmethod
    def seg(cls, df_reponse="nlp_seg"):
        nlp = cls(url, payload, df_reponse)
        return nlp


def init_df():
    df = pd.read_excel()
    return df


if __name__ == '__main__':
    df = init_df()
    df = nlp.run(df, "用户问")
    print(df[:5])
