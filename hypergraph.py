import torch
from torch_geometric.nn.conv import HypergraphConv

def apply_hypergraph_conv(x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None, out_channels=16):
    
    """
    하이퍼그래프 컨볼루션 네트워크 (HGCN) 레이어를 적용합니다.

    Contribution: HGCN은 고차원 관계를 모델링하여 노드 표현 학습을 강화하는 데 기여합니다.
    Input: 입력은 노드 특징과 하이퍼엣지 구조 정보(선택적으로 하이퍼엣지 가중치 및 특징 포함)이며,
    Output: 출력은 고차원 관계 정보가 통합된 새로운 노드 임베딩입니다.

    Args:
        x (torch.Tensor): (num_nodes, in_channels) 형태의 입력 노드 특징입니다.
        hyperedge_index (torch.Tensor): (2, num_edges_total) 형태의 하이퍼엣지 연결 정보입니다.
                                         첫 번째 행은 노드 인덱스를, 두 번째 행은 해당 노드가 속한 하이퍼엣지 인덱스를 포함합니다.
        hyperedge_weight (torch.Tensor, optional): (num_edges,) 형태의 각 하이퍼엣지에 대한 가중치입니다.
                                                    기본값은 None입니다.
        hyperedge_attr (torch.Tensor, optional): (num_edges, in_channels) 형태의 각 하이퍼엣지 자체의 속성 벡터입니다.
                                                  기본값은 None입니다.
        out_channels (int, optional): 원하는 출력 특징 차원입니다. 기본값은 16입니다.

    Returns:
        torch.Tensor: HGCN 레이어를 통과한 후의 (num_nodes, out_channels) 형태의 출력 노드 임베딩입니다.
    """
    
    in_channels = x.size(1)
    # HypergraphConv 레이어를 초기화합니다. 어텐션(attention) 기능과 드롭아웃(dropout), 헤드(heads)를 사용합니다.
    conv = HypergraphConv(in_channels, out_channels, use_attention=True, dropout=0.2, heads=2)
    # conv 레이어의 forward 메서드를 호출하여 하이퍼그래프 컨볼루션을 수행합니다.
    out = conv(x, hyperedge_index, hyperedge_weight=hyperedge_weight, hyperedge_attr=hyperedge_attr)
    return out

if __name__ == "__main__":
    # 1. 더미 주가 데이터 생성 및 전처리
    num_stocks = 3       # 주식 개수
    lookback = 4         # 과거 데이터 시점 수
    num_features = 5     # 각 시점의 특징 개수

    # 주가 데이터: (3, 4, 5) → 각 주식의 LOOKBACK과 FEATURE를 flatten하여 (3, 20)
    stock_data = torch.arange(num_stocks * lookback * num_features).reshape(num_stocks, lookback, num_features).float()

    # 각 주가에 대해 lookback과 features를 합치는데, 이때 순서를 지키면서 들어갑니다.
    x = stock_data.view(num_stocks, -1)

    print("x.shape:", x.shape)
    # 예상 결과: torch.Size([3, 20])

    # 2. 새로운 relation_data (하이퍼엣지 정보)
    # 각 행: 노드(주식), 각 열: 하이퍼엣지 (관계) = Stocks x Relations (섹터, 산업 등)
    relation_data = torch.tensor([
        [0, 1, 1, 1, 1],  # 주식 0이 속하는 하이퍼엣지 정보 (하이퍼엣지 1, 2, 3, 4에 속함)
        [1, 0, 1, 0, 0],  # 주식 1이 속하는 하이퍼엣지 정보 (하이퍼엣지 0, 2에 속함)
        [0, 0, 1, 0, 0]   # 주식 2가 속하는 하이퍼엣지 정보 (하이퍼엣지 2에 속함)
    ])
    print("relation_data:\n", relation_data)

    # 3. relation_data로부터 hyperedge_index 구성
    # nonzero 인덱스를 이용하여 hyperedge_index 생성
    # 첫 번째 행: 노드 인덱스, 두 번째 행: 해당 노드가 속한 하이퍼엣지 인덱스
    node_indices, hyperedge_indices = relation_data.nonzero(as_tuple=True)
    hyperedge_index = torch.stack([node_indices, hyperedge_indices], dim=0)
    print("hyperedge_index:\n", hyperedge_index)

    # hyperedge_index의 두 번째 행의 최댓값에 1을 더하면 총 하이퍼엣지 개수가 됩니다.
    num_edges = int(hyperedge_index[1].max().item()) + 1
    print("Number of hyperedges:", num_edges)
    # 예상 결과: 5 (인덱스 0~4)

    # 4. hyperedge_weight 재설정
    # 각 하이퍼엣지에 대한 가중치 (예시: 모든 하이퍼엣지에 동일한 가중치 1.0 사용)
    # 주어진 hyperedge_weight (하이퍼엣지 개수 = 5)
    hyperedge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    # 예시로 각 인덱스를 산업 이름에 매핑합니다.
    industry_names = ["기술", "헬스", "파이낸스", "통신", "축산"]

    # 5. hyperedge_attr 생성
    # 하이퍼엣지 특징은 (num_edges, in_channels) 크기로 만듭니다.
    # 주의: in_channels는 노드 특성의 차원과 동일해야 합니다.
    in_channels = x.size(1)  # 노드 특징의 차원 (여기서는 20)
    # 각 하이퍼엣지(산업)에 대한 무작위 특징 벡터를 생성합니다.
    hyperedge_attr = torch.rand(num_edges, in_channels)
    print("hyperedge_attr shape:", hyperedge_attr.shape)
    # 예상 결과: (5, 20)
    # hyperedge_attr은 엣지(산업)에 대한 특징 정보가 필요합니다 (예: 산업별 평균 성장률).
    # CI-STHPAN에서는 DTW로 정의된 (주식, 관계)를 (관계, 주식) 형태로 바꾸고, 이것을
    # x (Stocks, Features or Model Hidden Dimension)와 행렬 곱셈을 통해 만듭니다.

    # 6. HypergraphConv 레이어 (attention 기능 활성화) 정의 및 실행
    # 원하는 출력 차원
    output_channels = 16

    # 정의된 apply_hypergraph_conv 함수를 호출하여 하이퍼그래프 컨볼루션을 적용합니다.
    # 이때 hyperedge_weight와 hyperedge_attr 모두 전달합니다.
    output = apply_hypergraph_conv(x, hyperedge_index, hyperedge_weight=hyperedge_weight,
                                   hyperedge_attr=hyperedge_attr, out_channels=output_channels)
    print("Output shape:", output.shape)
    # 예상 결과: (3, 16)


'''
x.shape: torch.Size([3, 20])
relation_data:
 tensor([[0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]])
hyperedge_index:
 tensor([[0, 0, 0, 0, 1, 1, 2],
        [1, 2, 3, 4, 0, 2, 2]])
Number of hyperedges: 5
hyperedge_attr shape: torch.Size([5, 20])
Output shape: torch.Size([3, 32])
'''

