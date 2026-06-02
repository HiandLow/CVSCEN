import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import utils

def evaluate_selection_performance(selected_indices, ground_truth_indices):
    """
    선택된 변수 인덱스와 실제 정답 인덱스를 비교하여 FDR과 TPR을 계산합니다.
    """
    # 입력을 set으로 변환하여 집합 연산 수행
    selected = set(selected_indices)
    actual = set(ground_truth_indices)
    
    # 1. True Positives (TP): 정답인데 선택함
    tp = len(selected.intersection(actual))
    
    # 2. False Positives (FP): 정답이 아닌데 선택함 (Type I Error)
    fp = len(selected.difference(actual))
    
    # 3. False Negatives (FN): 정답인데 선택 안 함 (Type II Error)
    fn = len(actual.difference(selected))
    
    # TPR (True Positive Rate) = Recall = TP / (TP + FN)
    # 실제 정답 중 몇 개나 맞췄는가?
    tpr = tp / len(actual) if len(actual) > 0 else 0.0
    
    # FDR (False Discovery Rate) = FP / (TP + FP)
    # 선택된 것들 중 가짜(오답)의 비율은 얼마인가?
    total_selected = len(selected)
    fdr = fp / total_selected if total_selected > 0 else 0.0
    
    return (fdr, tpr)

def perform_post_hoc_selection(model, data_dict, alpha=0.05):
    """
    학습된 모델에 대해 사후 변수 선택을 진행합니다.
    """
    X = data_dict['x']
    T = data_dict['a']
    n_samples, n_features = X.shape
    
    # 1. 원본 데이터에 대한 예측 결과 생성
    # 특정 treatment point에서의 예측을 비교하거나, 평균적인 예측 차이를 볼 수 있습니다.
    # 여기서는 데이터에 존재하는 실제 T 값을 사용합니다.
    original_preds = model.predict(X, T)
    
    p_values = []
    
    # 2. 각 차원(feature)별로 무작위 섞기 진행
    for i in range(n_features):
        X_shuffled = X.copy()
        np.random.shuffle(X_shuffled[:, i])
        
        # 섞인 데이터에 대한 예측
        shuffled_preds = model.predict(X_shuffled, T)
        
        # 3. Wilcoxon Signed-Rank Test 진행 (원본 vs 섞은 결과)
        # 두 예측치 사이의 차이가 유의미한지 확인
        try:
            stat, p_val = wilcoxon(original_preds, shuffled_preds)
            if np.isnan(p_val):
                p_values.append(1.0)
            else:
                p_values.append(p_val)
        except ValueError:
            # 모든 값이 동일하여 rank를 계산할 수 없는 경우 (변화가 전혀 없는 경우)
            p_values.append(1.0)

        print(f"Feature {i}: p-value = {p_val:.4f}")

    # 4. FDR (False Discovery Rate) 계산 (Benjamini-Hochberg procedure)
    rejected, pvals_corrected = fdrcorrection(p_values, alpha=alpha)
    
    # 결과 정리
    selected_indices = np.where(rejected)[0]
    
    return {
        "selected_indices": selected_indices,
        "p_values": np.array(p_values),
        "p_values_corrected": pvals_corrected,
        "selection_mask": rejected
    }

# 실행 예시 (main 블록에 추가 가능)
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ihdp', choices=['ihdp', 'synt'])
args = parser.parse_args()
dataset_type = args.dataset

if __name__ == "__main__":
    from compare import LDML, CFDML, LRL, Base
    
    # 설정
    num_features = 25 if dataset_type == 'ihdp' else 100
    ground_truth_range = range(15) 

    c_uv = np.zeros(num_features, dtype=int)
    metrics_list = [] # 결과를 담을 리스트

    for i in range(10):
        # 1. f-string 수정
        try:
            data_name = f'./data/ihdp_semi_{i}.pkl' if dataset_type == 'ihdp' else f'./data/cont_synthetic_{i}.pkl'
            with open(data_name, 'rb') as f:
                loaded_data = pickle.load(f)
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {data_name}")
            continue
        
        # 2. 모델 학습
        test_model = Base() if dataset_type == 'ihdp' else LDML()
        test_model.fit(loaded_data['x'], loaded_data['a'], loaded_data['yf'])
        
        # 3. 변수 선택 실행
        result = perform_post_hoc_selection(test_model, loaded_data)
        
        # 4. 결과 누적
        c_uv[result['selected_indices']] += 1
        
        # 5. 성능 평가 결과 저장 (튜플 형태 (fdr, tpr))
        perf = evaluate_selection_performance(result['selected_indices'], ground_truth_range)
        metrics_list.append(perf)

    # 6. 평균 계산
    if metrics_list:
        avg_metrics = np.mean(metrics_list, axis=0)
        utils.plot_curve(num_features, c_uv, np.zeros_like(c_uv), str(np.sum(c_uv)), "0", "LDML_Union")
        print(f"\n--- 최종 결과 (10개 데이터셋 평균) ---")
        print(f"FDR: {avg_metrics[0]:.4f}, TPR: {avg_metrics[1]:.4f}")