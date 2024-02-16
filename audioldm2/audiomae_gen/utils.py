import torch.nn as nn

"""
utils.py 파일은 Prenet이라는 클래스를 정의. 
이 클래스는 입력 데이터를 변환하는 데 사용되며, 여러 선형 층과 ReLU 활성화 함수를 통해 데이터를 처리.
Dropout도 포함되어 있어 모델이 과적합을 방지할 수 있도록 도움.
"""

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128], dropout_rate=0.5):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_size, out_size)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


if __name__ == "__main__":
    model = Prenet(in_dim=128, sizes=[256, 256, 128])
    import ipdb

    ipdb.set_trace()
