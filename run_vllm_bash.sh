SCRIPT_DIR=$(cd $(dirname $0); pwd)
CACHEDIR=${SCRIPT_DIR}/vllm_root_cache
GPUOPTION="--gpus all"
IMAGE_NAME=vllm/vllm-openai:latest
OPTIONS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cpu)
            IMAGE_NAME="vllm-cpu-env"
            GPUOPTION=" "
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

docker run -it --rm ${GPUOPTION} \
  -v ${CACHEDIR}:/root/.cache \
  -v ${SCRIPT_DIR}/vllm:/vllm \
  -v ${SCRIPT_DIR}:/userdir -w /userdir \
  --name vllm_bash \
  --entrypoint="" ${IMAGE_NAME} bash