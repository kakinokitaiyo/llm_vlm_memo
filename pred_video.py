
try:
    import av
except:
    import pip, site, importlib
    pip.main(['install', 'av'])
    importlib.reload(site) 
    import av

from huggingface_hub import hf_hub_download
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Load the model in half-precision
model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

# ここが動画のpath
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")

conversation = [
    {

        "role": "user",
        "content": [
            {"type": "video", "path": video_path},
            {"type": "text", "text": "Why is this video funny?"}, # ここのtextの中身が質問文
            ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    num_frames=8,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, torch.float16)

out = model.generate(**inputs, max_new_tokens=60)
ret = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(ret[0])