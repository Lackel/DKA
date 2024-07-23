export CUDA_VISIBLE_DEVICES=0
python main.py \
    --dataset ok_vqa \
    --evaluation_set val \
    --train_annotations_path annotations/ok_vqa/train_annots_fixed.csv \
    --val_annotations_path annotations/ok_vqa/val_annots_fixed.csv \
    --test_annotations_path None \
    --train_images_dir /slurm-files/data/train2014/ \
    --val_images_dir /slurm-files/data/val2014/ \
    --test_images_dir None \
    --n_shots 10 \
    --k_ensemble 5 \
    --no_of_captions 9 \
    --mcan_examples_path mcan_examples/ok_vqa/examples.json \
    --llama_path /slurm-files/model/Llama-2-13b-hf \
    --blip_path /slurm-files/model/blip-itm-base-coco \
    --train_captions_path pnp_captions/ok_vqa/train_data_qr_captions.csv \
    --val_captions_path pnp_captions/ok_vqa/val_data_qr_captions.csv \
    --test_captions_path None \
    --decomposed_caption_path decomposed_caption/ok_vqa_val_dcaption.json \
    --decomposed_knowledge_path decompoased_knowledge/ok_vqa_val_dknowledge.json \
    --path_to_save_preds results/ok_vqa_val_with_mcan_llama2.csv

# python main.py \
#     --dataset a_ok_vqa \
#     --evaluation_set test \
#     --train_annotations_path annotations/a_ok_vqa/a_ok_vqa_train_fixed_annots.csv \
#     --val_annotations_path  annotations/a_ok_vqa/a_ok_vqa_val_fixed_annots.csv \
#     --test_annotations_path annotations/a_ok_vqa/a_ok_vqa_test_fixed_annots.csv \
#     --train_images_dir /slurm-files/data/train2017/ \
#     --val_images_dir /slurm-files/data/val2017/ \
#     --test_images_dir /slurm-files/data/test2017/ \
#     --n_shots 10 \
#     --k_ensemble 5 \
#     --no_of_captions 5 \
#     --mcan_examples_path mcan_examples/a_ok_vqa/examples_aokvqa_test.json \
#     --llama_path /slurm-files/model/Llama-2-13b-hf \
#     --blip_path /slurm-files/model/blip-itm-base-coco \
#     --train_captions_path pnp_captions/a_ok_vqa/a_ok_vqa_train_qr_captions.csv \
#     --val_captions_path pnp_captions/a_ok_vqa/a_ok_vqa_val_qr_captions.csv \
#     --test_captions_path pnp_captions/a_ok_vqa/a_ok_vqa_test_qr_captions.csv \
#     --decomposed_caption_path decomposed_caption/aokvqa_test_dcaption.json \
#     --decomposed_knowledge_path decompoased_knowledge/aokvqa_test_dknowledge.json \
#     --path_to_save_preds results/aok_vqa_val_with_mcan_llama2.csv
