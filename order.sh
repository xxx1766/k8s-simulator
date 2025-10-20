#!/bin/bash
# # 500 nodes with 100 MB/s bandwidth and 500jobs
# cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_500 
# cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 500 --factor 200 --bw 100 --node 500 && bash test.sh b 100 500 500  
# cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_500 
# cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 500 --factor 200 --bw 100 --node 500 && bash test.sh l 100 500 500
# cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_500
# cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 500 --factor 200 --bw 100 --node 500 && bash test.sh i 100 500 500
# 定义参数数组
NODES=(500)  # 如果要包含1000节点的测试，改为 NODES=(500 1000)
BANDWIDTHS=(100 50 150)
JOBS=(100 300 500)
SCHEDULERS=("bundle" "layer" "image")

# 调度器类型缩写映射
declare -A SCHEDULER_ABBR
SCHEDULER_ABBR["bundle"]="b"
SCHEDULER_ABBR["layer"]="l"
SCHEDULER_ABBR["image"]="i"

FACTOR=250

for nodes in "${NODES[@]}"; do
    for bw in "${BANDWIDTHS[@]}"; do
        for jobs in "${JOBS[@]}"; do
            echo "========================================"
            echo "Testing: ${nodes} nodes, ${bw} MB/s bandwidth, ${jobs} jobs"
            echo "========================================"
            
            for scheduler in "${SCHEDULERS[@]}"; do
                abbr="${SCHEDULER_ABBR[$scheduler]}"
                
                echo "Running ${scheduler} scheduler..."
                
                cd ~/kube-scheduler-simulator && \
                make docker_build_scheduler_${scheduler} docker_clean_daemons docker_up_local_${scheduler}_${jobs}
                
                cd ~/k8s-simulator && \
                bash initstore.sh ${nodes} && \
                python3.11 ${scheduler}Test.py --test ${jobs} --factor ${FACTOR} --bw ${bw} --node ${nodes} && \
                bash test.sh ${abbr} ${bw} ${nodes} ${jobs}
                
                echo "${scheduler} scheduler completed."
                echo "----------------------------------------"
            done
        done
    done
done

echo "All tests completed!"