#!/usr/bin/python3
#キャリブレーションしたファイルから読み込む
#精度はros2のものよりいい

import cv2
import numpy as np
from pupil_apriltags import Detector

# --- 設定項目 ---
# ステップ1で計測したマーカーの1辺の長さ（メートル単位）
TAG_SIZE_METERS = 0.077  # ★★★★★ あなたが計測した値に書き換えてください ★★★★★

# ステップ2で生成したキャリブレーションファイルを読み込む
try:
    camera_matrix = np.load('camera_matrix.npy')
    dist_coeffs = np.load('dist_coeffs.npy')
except FileNotFoundError:
    print("エラー: キャリブレーションファイルが見つかりません。")
    print("calibrate_camera.py を実行してファイルを作成してください。")
    exit()

# AprilTag検出器の初期化
at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0)

# カメラを起動
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # AprilTagを検出
    tags = at_detector.detect(gray, estimate_tag_pose=True,
                              camera_params=[camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]],
                              tag_size=TAG_SIZE_METERS)

    for tag in tags:
        # 検出したタグを緑の枠で囲む
        pts = tag.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # 検出結果を画面に描画
        pos_x = tag.pose_t[0][0]
        pos_y = tag.pose_t[1][0]
        pos_z = tag.pose_t[2][0]
        
        cv2.putText(frame, f"ID: {tag.tag_id}",
                    (tag.center.astype(int)[0], tag.center.astype(int)[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"X: {pos_x:.2f} m",
                    (tag.center.astype(int)[0], tag.center.astype(int)[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Y: {pos_y:.2f} m",
                    (tag.center.astype(int)[0], tag.center.astype(int)[1] + 0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Z: {pos_z:.2f} m",
                    (tag.center.astype(int)[0], tag.center.astype(int)[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("AprilTag Position Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
