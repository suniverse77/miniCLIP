# train for zero-shot model

# clip_weight: CLIP 모델 가중치 파일
# adapter_weight: Adaptor/Prompt 가중치 경로
# result_dir: 결과 저장 폴더
# data_dir: 데이터셋 경로
# dataset: 학습 데이터셋 (mvtec or visa)
# dataset_list: 평가 데이터셋 목록
# vis: 시각화 유무 (0이면 학습 및 평가 / 1이면 시각화)

# TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt
# TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt

run_name='small-alter-prompt2'

CUDA_VISIBLE_DEVICES=0 python main.py \
 --run_name ${run_name} \
 --mode train \
 --clip_weight ./weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt \
 --adapter_weight ./weight/${run_name} \
 --result_dir ./results/${run_name} \
 --data_dir ./data \
 --dataset mvtec \
 --dataset_list mvtec visa \
 --vis 0 \
 --epochs 12 \
 --batch_size 8 \
 --lr 0.0005 \
 --lambda1 3 \
 --lambda2 1 \
 --feature_layers 5 7 9 10
