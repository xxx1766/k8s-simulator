#!/bin/bash

# 初始化1000个节点的images.json文件
# 路径格式：/root/simulating/10.0.{nodeID//250}.{nodeID%250+1}/images.json

BASE_DIR="/root/simulating"
TOTAL_NODES=$1

mkdir -p "$BASE_DIR"
echo | tee $(docker inspect --format='{{.LogPath}}' layerdaemon)
echo | tee $(docker inspect --format='{{.LogPath}}' bundledaemon)
echo | tee $(docker inspect --format='{{.LogPath}}' imagedaemon)
echo | tee $(docker inspect --format='{{.LogPath}}' simulator-scheduler)
kubectl delete pods --all --force --grace-period=0

created_count=0
error_count=0
for ((nodeID=1; nodeID<=TOTAL_NODES; nodeID++)); do
    ip_part3=$((nodeID / 250))
    ip_part4=$((nodeID % 250 + 1))
    dir_path="$BASE_DIR/10.0.$ip_part3.$ip_part4"
    file_path="$dir_path/images.json"
    file_path_bundle="$dir_path/bundles.json"
    file_path_filejson="$dir_path/PrefabService/File.json"
    
    if mkdir -p "$dir_path"; then
        if echo '{}' > "$file_path"; then
            ((created_count++))
        fi
        if echo '{}' > "$file_path_bundle"; then
            ((created_count++))
        else
            echo "Error: Failed to create file $file_path"
            ((error_count++))
        fi
        if echo '{}' > "$file_path_filejson"; then
            ((created_count++))
        else
            echo "Error: Failed to create file $file_path_filejson"
            ((error_count++))
        fi
    else
        echo "Error: Failed to create directory $dir_path"
        ((error_count++))
    fi
done

echo "Initialization completed!"
