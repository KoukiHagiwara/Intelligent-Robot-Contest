#!/usr/bin/python3

from ultralytics import YOLO


model = YOLO("yolov11n.pt")# ファインチューニングの実行
model.train(
    data=yaml_path,
    epochs=100,     # 繰り返している回数
    imgsz=64,       # 入力画像のサイズを 64×64 ピクセルにリサイズする,CIFARに合わせて小さめ
    batch=32,       # 1回の更新に使う画像枚数
    name="ball_data_yolov11",
    save=True
)
