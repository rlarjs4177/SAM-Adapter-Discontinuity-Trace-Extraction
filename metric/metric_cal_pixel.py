import os
import numpy as np
import cv2

# GT 파일과 후처리 결과 파일을 비교하는 코드입니다.
# GT 폴더와 결과 폴더 경로를 지정하세요.

# 예시 경로 (필요에 따라 바꿔주세요)
gt_folder = './metric/gt/'
pred_folder = './metric/pred/'

# 파일 리스트 가져오기
gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(('.png', '.jpg'))])
pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(('.png', '.jpg'))])

# 파일마다 비교
total_files = min(len(gt_files), len(pred_files))
print(f"총 {total_files}개의 파일을 비교합니다.")

# 결과 저장용 변수
results = []

for idx, (gt_file, pred_file) in enumerate(zip(gt_files, pred_files)):
    # 파일 로딩
    gt_path = os.path.join(gt_folder, gt_file)
    pred_path = os.path.join(pred_folder, pred_file)

    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    # 이미지가 정상적으로 로딩됐는지 확인
    if gt is None:
        print(f"GT 파일 로딩 실패: {gt_path}")
        continue
    if pred is None:
        print(f"예측 파일 로딩 실패: {pred_path}")
        continue

    # 크기 확인 → 다르면 건너뜀
    if gt.shape != pred.shape:
        print(f"크기 불일치: {gt_file} vs {pred_file}, 건너뜀")
        continue

    # 픽셀 값이 이미 0/255라면 0/1로 변환, 아니라면 Threshold 적용
    if np.unique(gt).tolist() in ([0, 255], [0]):
        gt_bin = gt // 255
    else:
        _, gt_bin = cv2.threshold(gt, 128, 1, cv2.THRESH_BINARY)

    if np.unique(pred).tolist() in ([0, 255], [0]):
        pred_bin = pred // 255
    else:
        _, pred_bin = cv2.threshold(pred, 128, 1, cv2.THRESH_BINARY)

    # S-measure 계산
    alpha = 0.5
    gt_mean = np.mean(gt_bin)
    if gt_mean == 0:
        s_measure = 1 - np.mean(pred_bin)
    elif gt_mean == 1:
        s_measure = np.mean(pred_bin)
    else:
        object_score = np.mean(gt_bin * pred_bin) / (np.mean(gt_bin) + 1e-8)
        background_score = np.mean((1 - gt_bin) * (1 - pred_bin)) / (np.mean(1 - gt_bin) + 1e-8)
        s_measure = alpha * object_score + (1 - alpha) * background_score

    # Precision, Recall 계산
    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # F1, F2, Dice 계산
    adaptive_f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    beta_squared = 4
    f2_measure = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + 1e-8)
    dice = adaptive_f1

    results.append({
        "Image": idx + 1,
        "GT_File": gt_file,
        "Pred_File": pred_file,
        "S-Measure": s_measure,
        "Precision": precision,
        "Recall": recall,
        "Adaptive F1-Score": adaptive_f1,
        "F2-Measure": f2_measure,
        "Dice Coefficient": dice
    })

# 결과 출력
print("\n[개별 결과]")
for res in results:
    print(f"Image {res['Image']} ({res['GT_File']} vs {res['Pred_File']})")
    print(f"S-measure: {res['S-Measure']:.4f}")
    print(f"Precision: {res['Precision']:.4f}")
    print(f"Recall: {res['Recall']:.4f}")
    print(f"Adaptive F1-Score: {res['Adaptive F1-Score']:.4f}")
    print(f"F2-Measure: {res['F2-Measure']:.4f}")
    print(f"Dice Coefficient: {res['Dice Coefficient']:.4f}")
    print("-----------------------------")

# 비교 데이터 개수 출력
print(f"\n총 {len(results)}개의 이미지를 비교하였습니다.")

def metric_stats(metric_list):
    return np.mean(metric_list), np.min(metric_list), np.max(metric_list)

s_list = [r['S-Measure'] for r in results]
precision_list = [r['Precision'] for r in results]
recall_list = [r['Recall'] for r in results]
adaptive_f1_list = [r['Adaptive F1-Score'] for r in results]
f2_list = [r['F2-Measure'] for r in results]
dice_list = [r['Dice Coefficient'] for r in results]

mean_s, min_s, max_s = metric_stats(s_list)
mean_p, min_p, max_p = metric_stats(precision_list)
mean_r, min_r, max_r = metric_stats(recall_list)
mean_f1, min_f1, max_f1 = metric_stats(adaptive_f1_list)
mean_f2, min_f2, max_f2 = metric_stats(f2_list)
mean_dice, min_dice, max_dice = metric_stats(dice_list)

print("\n[최종 평균/최소/최대 결과]")
print(f"S-measure (구조적 유사도): 평균={mean_s:.4f}, 최소={min_s:.4f}, 최대={max_s:.4f}")
print(f"Adaptive F1-Score (beta=1): 평균={mean_f1:.4f}, 최소={min_f1:.4f}, 최대={max_f1:.4f}")
print(f"F2-measure (beta=2): 평균={mean_f2:.4f}, 최소={min_f2:.4f}, 최대={max_f2:.4f}")
print(f"Dice Coefficient: 평균={mean_dice:.4f}, 최소={min_dice:.4f}, 최대={max_dice:.4f}")
print(f"Precision: 평균={mean_p:.4f}, 최소={min_p:.4f}, 최대={max_p:.4f}")
print(f"Recall: 평균={mean_r:.4f}, 최소={min_r:.4f}, 최대={max_r:.4f}")

# 최소값이 나온 이미지 파일명 출력
def print_min_metric_img(metric_list, results, metric_name):
    min_value = np.min(metric_list)
    idx = metric_list.index(min_value)
    img_name = results[idx]['Pred_File']
    gt_name = results[idx]['GT_File']
    print(f"[{metric_name}] 최소값 {min_value:.4f} : {img_name} (GT: {gt_name})")
    
print("\n[최소값이 나온 이미지 파일명]")
print_min_metric_img(s_list, results, "S-measure")
print_min_metric_img(adaptive_f1_list, results, "Adaptive F1-Score")
print_min_metric_img(f2_list, results, "F2-measure")
print_min_metric_img(dice_list, results, "Dice Coefficient")
print_min_metric_img(precision_list, results, "Precision")
print_min_metric_img(recall_list, results, "Recall")
