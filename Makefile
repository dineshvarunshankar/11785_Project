IMAGE     = cgn_pytorch
CONTAINER = cgn_run

build:
	docker build \
	--build-arg BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
	--build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cu126 \
	-t $(IMAGE) .

build-cpu:
	docker build \
	--build-arg BASE=python:3.10-slim \
	--build-arg PYTORCH_INDEX=https://download.pytorch.org/whl/cpu \
	-t $(IMAGE)-cpu .

run:
	docker run --rm -it \
	--gpus all \
	-v $(PWD):/workspace \
	--name $(CONTAINER) \
	$(IMAGE)

run-cpu:
	docker run --rm -it \
	-v $(PWD):/workspace \
	--name $(CONTAINER) \
	$(IMAGE)-cpu

inference:
	docker run --rm -it \
	--gpus all \
	-v $(PWD):/workspace \
	$(IMAGE) python -m contact_graspnet_pytorch.inference \
	--np_path="test_data/*.npy" \
	--local_regions \
	--filter_grasps

test:
	docker run --rm -it \
	--gpus all \
	-v $(PWD):/workspace \
	$(IMAGE) python p1/pose_utils.py

verify:
	docker run --rm -it \
	--gpus all \
	-v $(PWD):/workspace \
	$(IMAGE) python p1/verify_pointnet2.py

down:
	docker stop $(CONTAINER) 2>/dev/null || true
	docker rm $(CONTAINER) 2>/dev/null || true

clean:
	docker rmi $(IMAGE) $(IMAGE)-cpu 2>/dev/null || true

.PHONY: build build-cpu run run-cpu inference test verify down clean
