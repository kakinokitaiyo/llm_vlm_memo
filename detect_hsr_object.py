import argparse
import cv2
import torch
from PIL import Image as PILImage
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except:
    import pip, site, importlib
    pip.main(['install', 'qwen_vl_utils'])
    importlib.reload(site)
    from qwen_vl_utils import process_vision_info

# --- HSR用画像取得ユーティリティ ---
class HSRUtil():
    def __init__(self, ri):
        self.ri = ri
        self.bridge = CvBridge()

    def getHeadRGBDImages(self, timeout=None):
        topic_names = ['/hsrb/head_rgbd_sensor/rgb/image_rect_color',
                       '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw',
                       '/hsrb/head_rgbd_sensor/depth_registered/camera_info']
        topic_types = [Image, Image, CameraInfo]
        sync_sub = self.ri.oneShotSyncSubscriber(topic_names, topic_types)
        sync_sub.waitResults(timeout=timeout)
        sub_data = sync_sub.data()
        if sub_data is not None:
            color_image = self.bridge.imgmsg_to_cv2(
                sub_data[0], desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(
                sub_data[1], desired_encoding='passthrough')
            camera_model = PinholeCameraModel()
            camera_model.fromCameraInfo(sub_data[2])
            return color_image, depth_image, camera_model
        else:
            return None, None, None

# --- メイン処理 ---
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, default="red can") 
    parser.add_argument('--image_path', type=str, default="cans.jpg") 
    parser.add_argument('--model_path', type=str, default="/userdir/vllm_root_cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/") 
    parser.add_argument('--output', type=str, default="") 
    args = parser.parse_args() 

    # HSRから画像取得
    exec(open('/choreonoid_ws/install/share/irsl_choreonoid/sample/irsl_import.py').read())
    ri = RobotInterface('robotinterface.yaml')
    hsr_util = HSRUtil(ri)
    color_img, _, _ = hsr_util.getHeadRGBDImages()
    pil_img = PILImage.fromarray(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

    # Qwen2-VL モデルとプロセッサの読み込み
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 入力メッセージ
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. When output a bounding box of the object in the image, the bounding box should start with <|box_start|> and end with <|box_end|>."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": f"Detect the bounding box of '{args.object}'."}
            ]
        }
    ]

    # Qwen2-VL 入力処理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to("cuda")

    # 推論
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text_with_tokens = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False)

    print("Model Output:", output_text_with_tokens[0])

    # 出力からバウンディングボックス抽出
    def crop_between(text, start, end):
        i1 = text.find(start)
        i2 = text.find(end)
        return text[i1+len(start):i2] if i1 != -1 and i2 != -1 else ""

    box_tokens = ['<|box_start|>', '<|box_end|>']
    if all(tok in output_text_with_tokens[0] for tok in box_tokens):
        box_str = crop_between(output_text_with_tokens[0], box_tokens[0], box_tokens[1])
        coords = [int(p.strip()) for p in box_str.replace("(", "").replace(")", "").split(',') if p.strip().isdigit()]
    else:
        coords = []

    # バウンディングボックス描画・保存
    if args.output and len(coords) >= 4:
        h, w = color_img.shape[:2]
        x1 = int(coords[0] / 1000 * w)
        y1 = int(coords[1] / 1000 * h)
        x2 = int(coords[2] / 1000 * w)
        y2 = int(coords[3] / 1000 * h)
        result_img = cv2.rectangle(color_img.copy(), (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.imwrite(args.output, result_img)
        print(f"Saved result to {args.output}")
    else:
        print("No valid bounding box found.")
