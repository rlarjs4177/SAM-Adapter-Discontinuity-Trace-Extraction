# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 14:29:55 2025

@author: USER
"""
import os
import numpy as np
import cv2

# GT 파일과 후처리 결과 파일을 비교하는 코드입니다.
# GT 폴더와 결과 폴더 경로를 지정하세요.

# 예시 경로 (필요에 따라 바꿔주세요)
gt_folder = './gt/'
pred_folder = './pred/'

# 파일 리스트 가져오기
gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(('.png', '.jpg'))])
pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(('.png', '.jpg'))])

# 파일마다 비교
total_files = min(len(gt_files), len(pred_files))
print(f"총 {total_files}개의 파일을 비교합니다.")

# 결과 저장용 변수
results = []

# 2x2 커널 설정
kernel_size = 2

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

    # 픽셀 값이 이미 0/255라면 0/1로 변환, 아니라면 Threshold 적용
    if np.unique(gt).tolist() in ([0, 255], [0]):
        gt_bin = gt // 255
    else:
        _, gt_bin = cv2.threshold(gt, 128, 1, cv2.THRESH_BINARY)

    if np.unique(pred).tolist() in ([0, 255], [0]):
        pred_bin = pred // 255
    else:
        _, pred_bin = cv2.threshold(pred, 128, 1, cv2.THRESH_BINARY)

    # 2x2 블록 단위로 비교
    h, w = gt_bin.shape
    h_blocks = h // kernel_size
    w_blocks = w // kernel_size

    block_tp = block_fp = block_fn = 0

    for i in range(h_blocks):
        for j in range(w_blocks):
            gt_block = gt_bin[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size]
            pred_block = pred_bin[i*kernel_size:(i+1)*kernel_size, j*kernel_size:(j+1)*kernel_size]

            gt_sum = np.sum(gt_block)
            pred_sum = np.sum(pred_block)

            if gt_sum > 0 and pred_sum > 0:
                block_tp += 1
            elif gt_sum == 0 and pred_sum > 0:
                block_fp += 1
            elif gt_sum > 0 and pred_sum == 0:
                block_fn += 1

    # Precision, Recall 계산
    precision = block_tp / (block_tp + block_fp + 1e-8)
    recall = block_tp / (block_tp + block_fn + 1e-8)

    # F1, F2, Dice 계산
    adaptive_f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    beta_squared = 4
    f2_measure = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + 1e-8)
    dice = adaptive_f1

    results.append({
        "Image": idx + 1,
        "GT_File": gt_file,
        "Pred_File": pred_file,
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
    print(f"Precision: {res['Precision']:.4f}")
    print(f"Recall: {res['Recall']:.4f}")
    print(f"Adaptive F1-Score: {res['Adaptive F1-Score']:.4f}")
    print(f"F2-Measure: {res['F2-Measure']:.4f}")
    print(f"Dice Coefficient: {res['Dice Coefficient']:.4f}")
    print("-----------------------------")

# 평균 계산
mean_precision = np.mean([r['Precision'] for r in results])
mean_recall = np.mean([r['Recall'] for r in results])
mean_adaptive_f1 = np.mean([r['Adaptive F1-Score'] for r in results])
mean_f2_measure = np.mean([r['F2-Measure'] for r in results])
mean_dice = np.mean([r['Dice Coefficient'] for r in results])

print("\n[최종 평균 결과]")
print(f"Adaptive F1-Score (beta=1): {mean_adaptive_f1:.4f}")
print(f"F2-measure (Adaptive F2 Score, beta=2): {mean_f2_measure:.4f}")
print(f"Dice Coefficient (F1 Score): {mean_dice:.4f}")
