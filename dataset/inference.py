#!/usr/bin/python3

import cv2
from ultralytics import YOLO

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ここに、あなたが学習させたモデルの正しいパスを指定してください！
# 例: 'runs/detect/train12/weights/best.pt'
model_path = 'runs/detect/train13/weights/best.pt' 
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# モデルを読み込む
model = YOLO(model_path)

# GPUが利用可能ならGPUを使う (Ultralyticsが自動で判断してくれます)
model.to('cuda')

# カメラを起動
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラを起動できませんでした。")
    exit()

# カメラの解像度を設定
cap.set(3, 640)
cap.set(4, 480)

while True:
    # カメラからフレームを1枚読み込む
    ret, img = cap.read()
    if not ret:
        print("エラー: フレームを読み込めませんでした。")
        break

    # モデルで推論を実行（信頼度が50%以上のものだけを検出）
    results = model.predict(img, conf=0.5)
    
    # 検出結果をフレームに描画
    annotated_frame = results[0].plot()

    # 結果を表示
    cv2.imshow('Real-time Ball Detection', annotated_frame)

    # 'ESC'キーが押されたらループを抜ける (キーコード 27 は ESC)
    k = cv2.waitKey(1)
    if k == 27:
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()