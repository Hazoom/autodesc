from typing import List

from bert_serving.client import BertClient

from scores.attentivepooling import AttentivePooler


class BertVectorsService:
    def __init__(self,
                 score_model_path: str,
                 ip: str = '127.0.0.1',
                 port: int = 5555,
                 port_out: int = 5556):
        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.pooller = AttentivePooler(score_model_path)

    def get_vectors(self,
                    sentence_list: List[str]):
        bc = BertClient(self.ip, self.port, self.port_out, show_server_config=False, check_length=False)
        vectors = bc.encode(sentence_list)
        return self.pooller.pool(vectors, sentence_list).tolist()


if __name__ == "__main__":
    bert_service = BertVectorsService('data/score_model/score_model.pkl', ip='3.248.147.179')
    vectors_demo = bert_service.get_vectors(sentence_list=['pandas dataframe merge two columns and merge two columns'])
    print(vectors_demo[0])
