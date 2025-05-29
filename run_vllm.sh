SCRIPT_DIR=$(cd $(dirname $0); pwd)
CACHEDIR=${SCRIPT_DIR}/vllm_root_cache
MODEL=Qwen/Qwen2-VL-2B-Instruct
SERVEDMODEL=Qwen2-VL-2B-Instruct
GPUOPTION="--gpus all"
IMAGE_NAME=vllm/vllm-openai:latest
OPTIONS=""
OPTIONS+=" --limit_mm_per_prompt image=8" # 同時に画像を8枚まで実行可能にする

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            case $2 in
                qwen2)
                    MODEL=Qwen/Qwen2-VL-2B-Instruct
                    SERVEDMODEL=Qwen2-VL-2B-Instruct
                    ;;
                qwen2-7b)
                    MODEL=Qwen/Qwen2-VL-7B-Instruct
                    SERVEDMODEL=Qwen2-VL-7B-Instruct
                    OPTIONS+=" --gpu_memory_utilization 0.9"
                    ;;
                qwen2.5)
                    MODEL=Qwen/Qwen2.5-VL-3B-Instruct
                    SERVEDMODEL=Qwen2.5-VL-3B-Instruct
                    OPTIONS+=" --max_model_len 87360"   ## 
                    ;;
                qwen2.5-7b)
                    MODEL=Qwen/Qwen2.5-VL-7B-Instruct
                    SERVEDMODEL=Qwen2.5-VL-7B-Instruct
                    OPTIONS+=" --max_model_len 87360"   ## 
                    ;;
                qwen2.5-32b)
                    MODEL=Qwen/Qwen2.5-VL-32B-Instruct
                    SERVEDMODEL=Qwen2.5-VL-32B-Instruct
                    OPTIONS+=" --max_model_len 87360"   ## 
                    ;;
                llava-1.5)
                    MODEL=llava-hf/llava-1.5-7b-hf
                    SERVEDMODEL=llava-1.5-7b-hf
                    OPTIONS+=" --chat-template /vllm/examples/template_llava.jinja --gpu_memory_utilization 0.95 "   ## VLAM 16GB設定 --max_model_len 1024
                    ;;
                llava-next-mistral)
                    MODEL=llava-hf/llava-v1.6-mistral-7b-hf
                    SERVEDMODEL=llava-v1.6-mistral-7b-hf
                    OPTIONS+=""   
                    ;;
                llava-next-vicuna)
                    MODEL=llava-hf/llava-v1.6-vicuna-7b-hf
                    SERVEDMODEL=llava-v1.6-vicuna-7b-hf
                    OPTIONS+=""   
                    ;;
                llava-next-video)
                    MODEL=llava-hf/LLaVA-NeXT-Video-7B-hf
                    SERVEDMODEL=LLaVA-NeXT-Video-7B-hf
                    OPTIONS+=""   
                    ;;
                blip2)
                    MODEL=Salesforce/blip2-opt-2.7b
                    SERVEDMODEL=blip2-opt-2.7b
                    OPTIONS+=" --chat-template /vllm/examples/template_blip2.jinja"
                    ;;
                blip2-6.7b)
                    MODEL=Salesforce/blip2-opt-6.7b
                    SERVEDMODEL=blip2-opt-6.7b
                    OPTIONS+=" --chat-template /vllm/examples/template_blip2.jinja"
                    ;;
                llava-onevision-qwen2)
                    MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf
                    SERVEDMODEL=llava-onevision-qwen2-0.5b-ov-hf
                    ;;
                llava-onevision-qwen2-7b)
                    MODEL=llava-hf/llava-onevision-qwen2-7b-ov-hf
                    SERVEDMODEL=llava-onevision-qwen2-7b-ov-hf
                    ;;
                # ultravox)
                #     MODEL=fixie-ai/ultravox-v0_5-llama-3_2-1b
                #     SERVEDMODEL=ultravox-v0_5-llama-3_2-1b
                #     ;;
                qwen2-audio-7b)
                    MODEL=Qwen/Qwen2-Audio-7B-Instruct
                    SERVEDMODEL=Qwen2-Audio-7B-Instruct
                    ;;
                deepseek-vl2-small)
                    MODEL=deepseek-ai/deepseek-vl2-small
                    SERVEDMODEL=deepseek-vl2-small
                    OPTIONS+=' --chat-template /vllm/examples/template_deepseek_vl2.jinja --hf-overrides '\{\"architectures\":\[\"DeepseekVLV2ForCausalLM\"\]\}''
                    ;;
                deepseek-vl2-tiny)
                    MODEL=deepseek-ai/deepseek-vl2-tiny
                    SERVEDMODEL=deepseek-vl2-tiny
                    OPTIONS+=' --chat-template /vllm/examples/template_deepseek_vl2.jinja --hf-overrides '\{\"architectures\":\[\"DeepseekVLV2ForCausalLM\"\]\}''
                    ;;
                *)
                    echo "Unknown model $2"
                    exit 1
                    ;;
            esac
            shift
            shift
            ;;
        -c|--cpu)
            IMAGE_NAME="vllm-cpu-env"
            GPUOPTION=" "
            shift
            ;;
        -o|--offload)
            OPTIONS+=" --cpu_offload_gb $2"
            shift
            shift
            ;;
        -g|--gpuid)
            GPUOPTION="--gpus device=$2"
            shift
            shift
            ;;
        --)
            shift
            break
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
        # *)
        #     POSITIONAL_ARGS+=("$1") # save positional arg
        #     shift # past argument
        #     ;;
    esac
done

set -x
docker run -it --rm \
  ${GPUOPTION}\
  --ipc=host --network=host \
  -v ${CACHEDIR}:/root/.cache \
  -v ${SCRIPT_DIR}/vllm:/vllm \
  ${IMAGE_NAME} \
  --model ${MODEL} --served-model-name ${SERVEDMODEL} ${OPTIONS}
  