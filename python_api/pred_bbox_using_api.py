import base64
from openai import OpenAI
import cv2


def np_to_base64(image):
    # 画像をPNG形式にエンコード
    _, buffer = cv2.imencode(".png", image)

    # バッファをBase64文字列に変換
    base64_bytes = base64.b64encode(buffer)
    base64_string = base64_bytes.decode("utf-8")
    return base64_string


def image_to_base64(image_path):
    # 画像を読み込む
    img = cv2.imread(image_path)
    # Base64変換
    return np_to_base64(img)


if __name__ == "__main__":
    openai_api_key = "sk-dummy"
    openai_api_base = "http://{}:{}/v1".format("localhost", 8000)

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id
    print(model)
    image_base64_1 = image_to_base64("cans.jpg")

    chat_completion_from_base64 = client.chat.completions.create(
        model=model,
        max_completion_tokens=256,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. When output a bounding box of the object in the image, the bounding box should start with <|box_start|> and end with <|box_end|>.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64_1}"},
                    },
                    {"type": "text", "text": "Detect the bounding box of red can."},
                ],
            },
        ],
        extra_body={"skip_special_tokens": False},
    )
    responce = chat_completion_from_base64.choices[0].message.content
    finish_reason = chat_completion_from_base64.choices[0].finish_reason
    print(responce, finish_reason)
