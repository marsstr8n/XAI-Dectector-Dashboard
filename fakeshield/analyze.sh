WEIGHT_PATH=./weight/fakeshield-v1-22b
IMAGE_PATH=./data/input.jpg
DTE_FDM_OUTPUT=./out/dte_fdm.jsonl
MFLM_OUTPUT=./out/mflm_output

pip install -q transformers==4.37.2  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
python -m llava.serve.cli \
    --model-path  ${WEIGHT_PATH}/DTE-FDM \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --image-path ${IMAGE_PATH} \
    --output-path ${DTE_FDM_OUTPUT}

pip install -q transformers==4.28.0  > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=0 \
python ./MFLM/cli_demo.py \
    --version ${WEIGHT_PATH}/MFLM \
    --DTE-FDM-output ${DTE_FDM_OUTPUT} \
    --MFLM-output ${MFLM_OUTPUT}
