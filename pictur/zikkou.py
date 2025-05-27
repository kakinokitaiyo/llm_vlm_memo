import cv2

# 入力画像の読み込み
image_path = "a.jpg"  # 入力画像ファイル名
image = cv2.imread(image_path)

# サイズを1000×1000にリサイズ
image = cv2.resize(image, (1000, 1000))

# バウンディングボックスの座標（左上 → 右下）
bbox_start = (425,351)
bbox_end = (715,876)

# 赤い枠を描く（BGRで (0, 0, 255)）
cv2.rectangle(image, bbox_start, bbox_end, color=(0, 0, 255), thickness=2)

# 結果をファイルに保存
output_path = "result2.jpg"
cv2.imwrite(output_path, image)

print(f"画像を {output_path} に保存しました。")
