#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

for i in {4..4}
do
    DATA_PATH="rice/demo$i"
    # DATA_PATH="wheat/demo1"
    TAG_BASE="rice_demo${i}_reg"  # 基础标签，可根据需要修改
    TAG="${TAG_BASE}_run0"
    OUTPUT_DIR="outputs/rice/nerfacto/$TAG"
    RENDER_DIR="render2/$TAG"
    CONFIG_FILE="$OUTPUT_DIR/config.yml"
    RENDER_OUTPUT="$RENDER_DIR/output_nerf_v21.json"

    echo "======== 第 $i 次训练开始 ========"
    
    # 创建必要的目录
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$RENDER_DIR"

    # 使用 Nerfacto 模型进行训练
    # echo "开始训练 Nerfacto 模型，输出目录: $OUTPUT_DIR"
    SECONDS=0
    ns-train nerfacto --data "$DATA_PATH" --output-dir "$OUTPUT_DIR"
    # echo "执行时间: $SECONDS 秒"

    # 训练完成后进行评估
    # echo "训练完成，开始评估并渲染结果到: $RENDER_DIR"
    ns-eval --load-config "$CONFIG_FILE" \
            --render-output-path "$RENDER_DIR" \
            --output-path "$RENDER_OUTPUT"

    # echo "开始导出 Nerfacto 点云，输出目录: $OUTPUT_DIR"
    # ns-export pointcloud --load-config "$CONFIG_FILE"  --output-dir "$RENDER_DIR"
    # ns-export pointcloud --load-config "$CONFIG_FILE" --output-dir exports/pcd/ --num-points 1000000 --remove-outliers True --normal-method open3d --save-world-frame False --obb_center 0.0000000000 0.0000000000 0.0000000000 --obb_rotation 0.0000000000 0.0000000000 0.0000000000 --obb_scale 1.0000000000 1.0000000000 1.0000000000
    # ns-export pointcloud --load-config "$CONFIG_FILE" --demo "$i" --output-dir "$RENDER_DIR" --num-points 1000000 --remove-outliers True --normal-method open3d --save-world-frame True --obb_center 0.0000000000 0.0000000000 0.0000000000 --obb_rotation 0.0000000000 0.0000000000 0.0000000000 --obb_scale 1.0000000000 1.0000000000 1.0000000000

    echo "======== 第 $i 次训练完成 ========"
    echo ""
done
