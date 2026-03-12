# DataLoader Guide
---

## Setup
Download the preprocessed data from [here](https://drive.google.com/drive/folders/14352W37usJVuVAsyby9LNksV68Gnmnfx?usp=drive_link)

Put the downloaded data in the data folder and create the splits folder.

```text
├── 11785_Project/
│   ├── data/
│       ├── splits/
│       │   ├── train.txt
│       │   ├── val.txt
│       │   └── test.txt
|       ├── preprocessed_data/
│       │   ├── scene_000000.npz
│       │   ├── scene_000001.npz
│       │   ├── ...
│       │   └── scene_008995.npz
└── 
```

This setup eliminates the need to download Acronym and ShapeNetSem, and it doesn't render the scene during the training for the dataloader, which should speed up training. 

Each scene_xxxxxx.npz file contains a dictionary, which has the following:
1. pc_cam: 
    * shape : (20000, 3)
    * desc  : 20k point clouds, input to the network
2. camera_pose:
    * shape : (4, 4)
    * desc  : SE(3) coordinates of camera position in the world frame, i.e. the camera pose
3. pos_contact_points:
    * shape : (16000, 3)
    * desc  : positive contact points, which have been collected for 8000 grasps
4. pos_contact_dirs:
    * shape : (16000, 3)
    * desc  : normalized contact baseline directions derived from each contact pair (scene_contact_points[:,0] - scene_contact_points[:,1] and opposite direction)
5. pos_finger_diffs:
    * shape : (16000,)
    * desc  : width of the grasp, i.e. the euclidean distance between the two grasp fingers
6. pos_approach_dirs:
    * shape : (16000, 3) 
    * desc  : It comes from grasp_transforms[:,:3,2] (grasp z/approach axis), then repeated per contact and normalized. 

there are 16000 positive contact points, which have been collected for 8000 grasps. this value can be changed in the config.yaml file, but both the pytorch_CGN and the original CGN use this value.

The dataloader will load a scene and return all the information mentioned above as a dictionary with __getitem__ method.

The keys for the dict from dataloader must exactly be: pc_cam, camera_pose, pos_contact_points, pos_contact_dirs, pos_finger_diffs, pos_approach_dirs

You can visualize one of the preprocessed scene with the docker using the following command: (though I am not sure if this works for everybody)
```
xhost +local:docker 2>/dev/null || true
docker run --rm -it \
  --gpus all \
  --user $(id -u):$(id -g) \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=/tmp \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  cgn_pytorch python p2/visualize_preprocessed_scene.py \
  --scene-path data/preprocessed_data/scene_000000.npz \
  --show
```

alternatively, I used a conda env to write the scripts and used rener_scene_rgbd.py, which opens your browser to inspect the data. 

and you can visualize using: 
```
python3 p2/render_scene_rgbd.py --scene-index 2 --overlay-contacts
```