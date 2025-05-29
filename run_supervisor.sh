SCRIPT_DIR=$(cd $(dirname $0); pwd)
GPUOPTION="--gpus all"


while [[ $# -gt 0 ]]; do
    case $1 in
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
    esac
done

set -x

docker run --rm -d \
--net=host \
${GPUOPTION} \
-v ${SCRIPT_DIR}/vllm_root_cache:/root/.cache \
-v ${SCRIPT_DIR}/supervisor_configs:/etc/supervisor/conf.d \
--entrypoint="" \
--name=vllm-openai-supervisor \
vllm-openai-supervisor /usr/bin/supervisord   

