from utils.df_async_post import DfAsyncPost
import pandas as pd

url = ""
payload = "{\"query\":\"{}\"}"


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