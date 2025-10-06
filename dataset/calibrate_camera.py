#!/usr/bin/python3

import cv2
import numpy as np
import glob

# チェスボードの設定
CHESSBOARD_CORNERS_X = 9  # チェスボードの内部の角の数（横）
CHESSBOARD_CORNERS_Y = 6  # チェスボードの内部の角の数（縦）

# 3D世界の座標を準備 (0,0,0), (1,0,0), ...
objp = np.zeros((CHESSBOARD_CORNERS_X * CHESSBOARD_CORNERS_Y, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS_X, 0:CHESSBOARD_CORNERS_Y].T.reshape(-1, 2)

# 画像から検出された3D点と2D点を保存する配列
objpoints = []  # 3D点
imgpoints = []  # 2D点

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

print("チェスボードをカメラに向けてください。")
print("いろんな角度から見せて、'c'キーを押して画像をキャプチャします。")
print("15枚以上集まったら 'q' キーを押してキャリブレーションを開始します。")

img_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # チェスボードの角を探す
    ret_corners, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_X, CHESSBOARD_CORNERS_Y), None)

    # 見つかったら描画
    if ret_corners:
        cv2.drawChessboardCorners(frame, (CHESSBOARD_CORNERS_X, CHESSBOARD_CORNERS_Y), corners, ret_corners)

    cv2.putText(frame, f"Captured: {img_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Calibration', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and ret_corners:
        objpoints.append(objp)
        imgpoints.append(corners)
        img_count += 1
        print(f"画像をキャプチャしました: {img_count}枚目")
    elif key == ord('q'):
        if img_count < 15:
            print("画像が少なすぎます。15枚以上キャプチャしてください。")
        else:
            break

cap.release()
cv2.destroyAllWindows()

print("キャリブレーションを実行中...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("キャリブレーションに成功しました！")
    # 結果をファイルに保存
    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)
    print("カメラ行列を 'camera_matrix.npy' として保存しました。")
    print("歪み係数を 'dist_coeffs.npy' として保存しました。")
else:
    print("キャリブレーションに失敗しました。")
