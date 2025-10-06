#!/usr/bin/python3

import cv2
import numpy as np
from pupil_apriltags import Detector

# --- 設定項目 ---
# ステップ1で計測したマーカーの1辺の長さ（メートル単位）
TAG_SIZE_METERS = 0.077  # ★★★★★ あなたが計測した値に書き換えてください ★★★★★

# --- ここからカメラパラメータの手動設定 ---
# 別の端末のキャリブレーションで得た焦点距離
FOCAL_LENGTH_X = 839.6533065
FOCAL_LENGTH_Y = 839.6533065

# カメラの解像度 (お使いのカメラに合わせて設定)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 画像の中心座標 (通常は解像度の半分)
CENTER_X = FRAME_WIDTH / 2
CENTER_Y = FRAME_HEIGHT / 2

# AprilTag検出器に渡すカメラパラメータのリストを作成
camera_params = [FOCAL_LENGTH_X, FOCAL_LENGTH_Y, CENTER_X, CENTER_Y]
# --- ここまで ---


# AprilTag検出器の初期化
at_detector = Detector(families='tag36h11', nthreads=1, quad_decimate=1.0)

# カメラを起動
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

# カメラの解像度を設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # AprilTagを検出 (camera_paramsをここで渡す)
    tags = at_detector.detect(gray, estimate_tag_pose=True,
                              camera_params=camera_params,
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
