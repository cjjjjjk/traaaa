import pandas as pd
import os

# configuration
DATASET_DIR = 'dataset'
FEATURES_CSV = os.path.join(DATASET_DIR, 'label_cv2_v2.csv')
LABEL_CSV = os.path.join(DATASET_DIR, 'label.csv')

def compute_congestion_score(
    edge_density,
    contour_area_ratio,
    orientation_entropy,
    mean_nn_dist_norm,
    local_density
):
    """
    tính congestion score dựa trên features
    
    thresholds:
    - 0.0 - 0.4: không tắc / tắc nhẹ
    - 0.4 - 0.65: đông nhưng không tắc / tắc vừa
    - 0.65 - 0.99: tắc nghẽn nghiêm trọng
    """
    score = 0.0

    # 1. lượng phương tiện
    if edge_density > 0.22:
        score += 0.25
    elif edge_density > 0.18:
        score += 0.20
    elif edge_density > 0.14:
        score += 0.12
    else:
        score += 0.05

    # 2. mức chen chúc (rất quan trọng)
    if contour_area_ratio > 0.30:
        score += 0.25
    elif contour_area_ratio > 0.22:
        score += 0.18
    elif contour_area_ratio > 0.17:
        score += 0.10
    else:
        score += 0.05

    # 3. khoảng cách phương tiện (trọng số cao nhất)
    if mean_nn_dist_norm < 0.050:
        score += 0.30
    elif mean_nn_dist_norm < 0.055:
        score += 0.22
    elif mean_nn_dist_norm < 0.060:
        score += 0.12
    else:
        score += 0.05

    # 4. mật độ cục bộ
    if local_density > 2.5:
        score += 0.12
    elif local_density > 2.0:
        score += 0.08
    elif local_density > 1.5:
        score += 0.04

    # 5. độ lộn xộn (chỉ tinh chỉnh)
    if orientation_entropy > 0.592:
        score += 0.05
    elif orientation_entropy > 0.588:
        score += 0.03

    return round(min(score, 0.99), 2)

# read features csv
df_features = pd.read_csv(FEATURES_CSV)
print(f"loaded {len(df_features)} records from {FEATURES_CSV}")

# compute congestion score for each row
print("\ncomputing congestion scores...")
scores = []

for idx, row in df_features.iterrows():
    score = compute_congestion_score(
        edge_density=row['edge_density'],
        contour_area_ratio=row['contour_area_ratio'],
        orientation_entropy=row['orientation_entropy'],
        mean_nn_dist_norm=row['mean_nn_dist_norm'],
        local_density=row['local_density']
    )
    scores.append(score)
    
    # print sample progress
    if (idx + 1) % 200 == 0:
        print(f"  processed {idx + 1}/{len(df_features)} images...")

# create label dataframe with image_name and congestion_score
df_label = df_features[['image_name']].copy()
df_label['congestion_score'] = scores

# save to label csv
df_label.to_csv(LABEL_CSV, index=False)

print(f"\n✓ saved {len(df_label)} labeled records to {LABEL_CSV}")

# show statistics
print("\ncongestion score statistics:")
print(df_label['congestion_score'].describe())

print("\nscore distribution:")
low = len(df_label[df_label['congestion_score'] < 0.4])
mid = len(df_label[(df_label['congestion_score'] >= 0.4) & (df_label['congestion_score'] < 0.65)])
high = len(df_label[df_label['congestion_score'] >= 0.65])

print(f"  không tắc / tắc nhẹ (< 0.4):        {low:4d} ({low/len(df_label)*100:.1f}%)")
print(f"  đông / tắc vừa (0.4 - 0.65):        {mid:4d} ({mid/len(df_label)*100:.1f}%)")
print(f"  tắc nghẽn nghiêm trọng (>= 0.65):   {high:4d} ({high/len(df_label)*100:.1f}%)")

print(f"\npreview:")
print(df_label.head(10))
