# Test Example 1 : (PreCls, Motif Model)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 10027 \
    --nproc_per_node=1 \
    tools/relation_test_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    TEST.IMS_PER_BATCH 1 \
    DTYPE "float16" \
    GLOVE_DIR /home/kaihua/glove \
    MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp \
    OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp