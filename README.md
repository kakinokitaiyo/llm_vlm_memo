# 準備
## ollama
```
docker pull ollama/ollama
```
## vllm
```
ghcr.io/open-webui/open-webui:main
```
### GPU using supervisor
```
./build_supervisor.sh
```

### GPU using terminal
```
docker pull vllm/vllm-openai:latest
```

### CPU using terminal
vllm/docker/Dockerfile.cpuに以下環境変数を追記する．
```ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.8.4```
```
cd vllm
docker build -f docker/Dockerfile.cpu --tag vllm-cpu-env --target vllm-openai .
cd ..
```

# 実行(サーバ - クライアント)
## サーバ側
### ollama
```
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```
### vllm using supervisor
```
./run_supervisor.sh
```
webブラウザで`http://127.0.0.1:9999`にアクセスして，使用したいモデルのstartを押す．(他サーバで立ち上げた場合にはアクセスするIPアドレスを変更する)

### vllm using terminal
以下適切なものを選ぶ．
いろいろ使っている感じとしてはQwen2-VL-7BとかQwen2.5-VL-7Bとかがいい感じに見える．
- Qwen2-VL-2B (GPU)
    ```
    ./run_vllm.sh
    ```
- Qwen2-VL-7B (CPU)
    ```
    ./run_vllm.sh --model qwen2-7b --cpu
    ```
- Qwen2-VL-7B (GPU)
    ```
    # メモリオフロードすることでGPUメモリが少なくても実行出来る
    # --offloadを使うと起動が遅くなる
    ./run_vllm.sh --model qwen2-7b --offload 8
    ```
- Qwen2.5-VL-3B (GPU)
    ```
    ./run_vllm.sh --model qwen2.5
    ```
- Qwen2.5-VL-7B (GPU)
    ```
    ./run_vllm.sh --model qwen2.5-7b  --offload 16
    ```
- llava-1.5-7b (GPU)
    ```
    ./run_vllm.sh --model llava-1.5
    ```
- blip2-2.7b (GPU)
    ```
    ./run_vllm.sh --model blip2
    ```
- blip2-6.7b (CPU)
    ```
    ./run_vllm.sh --model blip2-6.7b --cpu
    ```

## クライアント側
### CUI (ollama - llama3)
1. 端末で実行
```
exec_llama3.sh
```
### webUI (ollama, vllm)
1. 端末で実行
    ```
    docker run -d -p 5955:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
    ```
2. http://localhost:5955 にアクセスする．

### python
python_apiディレクトリを参考にする．

- ollama
以下を参考にする．
    - https://www.ollama.com/blog/openai-compatibility
- vllm
以下が参考になる．
    - https://qiita.com/engchina/items/f7cc32f3f34011b69a18#api%E3%83%AA%E3%82%AF%E3%82%A8%E3%82%B9%E3%83%88%E3%81%AE%E9%80%81%E4%BF%A1

# 実行(スタンドアローン)

## vllm

### video
```
./run_vllm_bash.sh
```
```
python3 pred_video.py
```
### image
```
./run_vllm_bash.sh
```
```
python3 pred_bbox.py --object "blue can" --image_path cans.jpg --output result.png
# OR
python3 pred_bbox2.py --object "blue can" --image_path cans.jpg --output result.png
```

