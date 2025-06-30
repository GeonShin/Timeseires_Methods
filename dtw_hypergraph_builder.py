import pandas as pd 
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import torch

'''
DTW (Dynamic Time Warping) 개요
DTW Input: 두 개의 시계열 데이터와 시계열 내 지점 간 거리를 측정하는 함수.
DTW Output: 두 시계열 간의 비선형적으로 정렬된 최소 거리 값과 최적의 정렬 경로.
DTW Contribution: 길이가 다르거나, 같은 패턴이라도 발생하는 시점이 밀리거나 당겨지는 경우에도 시계열 데이터 간의 정확한 유사도를 측정하여 다양한 패턴 인식 및 비교 분석에 활용됩니다.

코드의 목적
Input: 각 시장별 CSV 파일에서 추출된 주식들의 log_return(Exampe: OHLCV) 시계열 데이터와 DTW 및 하이퍼엣지 생성을 위한 설정 값들.
Output: 훈련 및 검증 기간에 대해 DTW 유사성을 기반으로 구축된 주식 간의 하이퍼엣지 인덱스를 담은 PyTorch 파일.
Contribution: 주식 시계열 데이터의 비선형적 유사도를 활용하여 동적인 고차원 관계(하이퍼엣지)를 효과적으로 구축, 이는 하이퍼그래프 컨볼루션(Pytorch)에 들어가는 input 데이터(그래프 정보)를 제공.
'''

# DTW 행렬을 하이퍼엣지 인덱스로 변환하는 함수
def dtw_to_hyperedge_index(dtw_matrix: np.ndarray, top_k: int = 15) -> torch.Tensor:
    """
    dtw_matrix: (num_stocks, num_stocks) shaped symmetric matrix from fastdtw
    top_k: number of closest nodes (excluding self) to include in each hyperedge

    returns:
        hyperedge_index: torch.Tensor of shape (2, num_edges), where
                            row 0 is stock indices,
                            row 1 is corresponding hyperedge indices.
    """
    num_stocks = dtw_matrix.shape[0]
    hyperedges = []

    # 각 주식에 대해 가장 가까운 top_k 주식 선택 (자기 자신 제외)
    for i in range(num_stocks):
        neighbors = np.argsort(dtw_matrix[i])  # 거리 순으로 정렬
        top_neighbors = neighbors[1:top_k+1]  # 자기 자신 제외하고 top_k 선택
        hyperedge = [i] + top_neighbors.tolist()  # 자기 자신 포함해서 하이퍼엣지 구성
        hyperedges.append(hyperedge)

    # 하이퍼엣지를 edge_index로 변환
    edge_list = []
    for hedge_id, hedge in enumerate(hyperedges):
        for node in hedge:
            edge_list.append([node, hedge_id])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # (2, num_edges)
    return edge_index


# 시장 이름 리스트
markets = ['RAY', 'RIY', 'RTY']

# 시장별로 데이터 처리
for market in markets:
    # CSV 파일 읽기 (다층 인덱스 처리)
    df = pd.read_csv(f'/data/shingeon/Stock-Library/Loader/features/{market}_all_russell_16_klNone_.csv',
                     index_col=[0, 1], 
                     header=[0, 1],
                     skipinitialspace=True)

    # 'log_return' feature만 추출하여 'feature'를 분리
    # 어떤 indicator를 선택할지 생각: (예: OHLCV)
    df = df['feature']['log_return'].unstack(1)

    # 유효기간과 테스트기간 설정
    valid_start_str = '2016-01-04' 
    test_start_str = '2017-01-03' 

    # 훈련, 검증 데이터 분할
    train_df = df.loc[:valid_start_str]
    valid_df = df.loc[valid_start_str:test_start_str]
    
    # 티커 수 계산
    no_tickers = len(df.columns)

    # DTW 계산 함수 정의
    def fastdtw_ij(df):
        # NaN 값을 0으로 채우기
        df.fillna(0, inplace=True)
        stock_npy = df.to_numpy()
        no_tickers = stock_npy.shape[1] 
        fastdtw_ij = np.zeros([no_tickers, no_tickers], dtype=np.float32)

        # 각 주식에 대해 DTW 거리 계산
        for i in range(no_tickers):
            X = stock_npy[:, i]  # i번째 주식 선택
            for j in range(i, no_tickers):
                Y = stock_npy[:, j]  # j번째 주식 선택
                x, y = np.expand_dims(X, axis=0), np.expand_dims(Y, axis=0)  # 2D 배열로 변환
                dtw, path = fastdtw(x, y, dist=euclidean)  # DTW 계산
                fastdtw_ij[i][j] = dtw
                fastdtw_ij[j][i] = dtw  # 대칭적인 DTW 거리 행렬 구성
        return fastdtw_ij

    # 훈련 데이터와 검증 데이터에 대해 DTW 거리 계산
    train_fastdtw_ij = fastdtw_ij(train_df)
    valid_fastdtw_ij = fastdtw_ij(valid_df)

    # 훈련 데이터와 검증 데이터에 대한 하이퍼엣지 인덱스 생성
    train_hyperedge_index = dtw_to_hyperedge_index(train_fastdtw_ij)
    valid_hyperedge_index = dtw_to_hyperedge_index(valid_fastdtw_ij)

    # 하이퍼엣지 인덱스를 딕셔너리로 저장하여 PyTorch 파일로 저장
    torch.save({
        "train_dtw": train_hyperedge_index,
        "valid_dtw": valid_hyperedge_index
    }, f"{market}_dtw_hyperedge_index.pt")
