# 500 nodes with 100 MB/s bandwidth and 500jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_500 
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 500 --factor 200 --bw 100 --node 500 && bash test.sh b 100 500 500  
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_500 
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 500 --factor 200 --bw 100 --node 500 && bash test.sh l 100 500 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_500
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 500 --factor 200 --bw 100 --node 500 && bash test.sh i 100 500 500

# 500 nodes with 100 MB/s bandwidth and 300jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_300
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 300 --factor 200 --bw 100 --node 500 && bash test.sh b 100 500 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_300
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 300 --factor 200 --bw 100 --node 500 && bash test.sh l 100 500 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_300
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 300 --factor 200 --bw 100 --node 500 && bash test.sh i 100 500 300

# 500 nodes with 100 MB/s bandwidth and 100jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_100
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 100 --factor 200 --bw 100 --node 500 && bash test.sh b 100 500 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_100
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 100 --factor 200 --bw 100 --node 500 && bash test.sh l 100 500 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_100
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 100 --factor 200 --bw 100 --node 500 && bash test.sh i 100 500 100

# 500 nodes with 50 MB/s bandwidth and 500jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_500 
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 500 --factor 200 --bw 50 --node 500 && bash test.sh b 50 500 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_500 
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 500 --factor 200 --bw 50 --node 500 && bash test.sh l 50 500 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_500
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 500 --factor 200 --bw 50 --node 500 && bash test.sh i 50 500 500

# 500 nodes with 50 MB/s bandwidth and 300jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_300
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 300 --factor 200 --bw 50 --node 500 && bash test.sh b 50 500 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_300
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 300 --factor 200 --bw 50 --node 500 && bash test.sh l 50 500 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_300
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 300 --factor 200 --bw 50 --node 500 && bash test.sh i 50 500 300  

# 500 nodes with 50 MB/s bandwidth and 100jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_100
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 100 --factor 200 --bw 50 --node 500 && bash test.sh b 50 500 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_100
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 100 --factor 200 --bw 50 --node 500 && bash test.sh l 50 500 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_100
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 100 --factor 200 --bw 50 --node 500 && bash test.sh i 50 500 100

# 500 nodes with 150 MB/s bandwidth and 500jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_500 
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 500 --factor 200 --bw 150 --node 500 && bash test.sh b 150 500 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_500 
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 500 --factor 200 --bw 150 --node 500 && bash test.sh l 150 500 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_500
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 500 --factor 200 --bw 150 --node 500 && bash test.sh i 150 500 500

# 500 nodes with 150 MB/s bandwidth and 300jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_300
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 300 --factor 200 --bw 150 --node 500 && bash test.sh b 150 500 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_300
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 300 --factor 200 --bw 150 --node 500 && bash test.sh l 150 500 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_300
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 300 --factor 200 --bw 150 --node 500 && bash test.sh i 150 500 300

# 500 nodes with 150 MB/s bandwidth and 100jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_100
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 bundleTest.py --test 100 --factor 200 --bw 150 --node 500 && bash test.sh b 150 500 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_100
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 layerTest.py --test 100 --factor 200 --bw 150 --node 500 && bash test.sh l 150 500 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_100
cd ~/k8s-simulator && bash initstore.sh 500 && python3.11 imageTest.py --test 100 --factor 200 --bw 150 --node 500 && bash test.sh i 150 500 100

# 1000 nodes with 100 MB/s bandwidth and 500jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_500
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 bundleTest.py --test 500 --factor 200 --bw 100 --node 1000 && bash test.sh b 100 1000 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_500
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 layerTest.py --test 500 --factor 200 --bw 100 --node 1000 && bash test.sh l 100 1000 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_500
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 imageTest.py --test 500 --factor 200 --bw 100 --node 1000 && bash test.sh i 100 1000 500 

# 1000 nodes with 100 MB/s bandwidth and 300jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_300
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 bundleTest.py --test 300 --factor 200 --bw 100 --node 1000 && bash test.sh b 100 1000 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_300
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 layerTest.py --test 300 --factor 200 --bw 100 --node 1000 && bash test.sh l 100 1000 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_300
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 imageTest.py --test 300 --factor 200 --bw 100 --node 1000 && bash test.sh i 100 1000 300

# 1000 nodes with 100 MB/s bandwidth and 100jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_100
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 bundleTest.py --test 100 --factor 200 --bw 100 --node 1000 && bash test.sh b 100 1000 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_100
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 layerTest.py --test 100 --factor 200 --bw 100 --node 1000 && bash test.sh l 100 1000 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_100
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 imageTest.py --test 100 --factor 200 --bw 100 --node 1000 && bash test.sh i 100 1000 100

# 1000 nodes with 50 MB/s bandwidth and 500jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_500 
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 bundleTest.py --test 500 --factor 200 --bw 50 --node 1000 && bash test.sh b 50 1000 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_500 
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 layerTest.py --test 500 --factor 200 --bw 50 --node 1000 && bash test.sh l 50 1000 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_500
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 imageTest.py --test 500 --factor 200 --bw 50 --node 1000 && bash test.sh i 50 1000 500

# 1000 nodes with 50 MB/s bandwidth and 300jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_300
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 bundleTest.py --test 300 --factor 200 --bw 50 --node 1000 && bash test.sh b 50 1000 300  
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_300
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 layerTest.py --test 300 --factor 200 --bw 50 --node 1000 && bash test.sh l 50 1000 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_300
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 imageTest.py --test 300 --factor 200 --bw 50 --node 1000 && bash test.sh i 50 1000 300

# 1000 nodes with 50 MB/s bandwidth and 100jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_100
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 bundleTest.py --test 100 --factor 200 --bw 50 --node 1000 && bash test.sh b 50 1000 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_100
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 layerTest.py --test 100 --factor 200 --bw 50 --node 1000 && bash test.sh l 50 1000 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_100
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 imageTest.py --test 100 --factor 200 --bw 50 --node 1000 && bash test.sh i 50 1000 100

# 1000 nodes with 150 MB/s bandwidth and 500jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_500 
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 bundleTest.py --test 500 --factor 200 --bw 150 --node 1000 && bash test.sh b 150 1000 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_500
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 layerTest.py --test 500 --factor 200 --bw 150 --node 1000 && bash test.sh l 150 1000 500
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_500
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 imageTest.py --test 500 --factor 200 --bw 150 --node 1000 && bash test.sh i 150 1000 500

# 1000 nodes with 150 MB/s bandwidth and 300jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_300
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 bundleTest.py --test 300 --factor 200 --bw 150 --node 1000 && bash test.sh b 150 1000 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_300
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 layerTest.py --test 300 --factor 200 --bw 150 --node 1000 && bash test.sh l 150 1000 300
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_300
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 imageTest.py --test 300 --factor 200 --bw 150 --node 1000 && bash test.sh i 150 1000 300

# 1000 nodes with 150 MB/s bandwidth and 100jobs
cd ~/kube-scheduler-simulator && make docker_build_scheduler_bundle docker_clean_daemons docker_up_local_bundle_100
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 bundleTest.py --test 100 --factor 200 --bw 150 --node 1000 && bash test.sh b 150 1000 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_layer docker_clean_daemons docker_up_local_layer_100
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 layerTest.py --test 100 --factor 200 --bw 150 --node 1000 && bash test.sh l 150 1000 100
cd ~/kube-scheduler-simulator && make docker_build_scheduler_image docker_clean_daemons docker_up_local_image_100
cd ~/k8s-simulator && bash initstore.sh 1000 && python3.11 imageTest.py --test 100 --factor 200 --bw 150 --node 1000 && bash test.sh i 150 1000 100


