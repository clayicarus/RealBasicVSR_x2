# RealBasicVSR_x2

A model for x2 real-world video super-resolution based on [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR).

## Installation
1. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/locally/), e.g.,
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

2. Install mim and mmcv-full
```
pip install openmim
mim install mmcv-full
```

3. Install mmedit
```
pip install mmedit
```

## Inference
1. Download the pre-trained weights to `checkpoints/`. ([Dropbox](https://www.dropbox.com/s/eufigxmmkv5woop/RealBasicVSR.pth?dl=0) / [Google Drive](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view) / [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/chan0899_e_ntu_edu_sg/EfMvf8H6Y45JiY0xsK4Wy-EB0kiGmuUbqKf0qsdoFU3Y-A?e=9p8ITR))

2. Run the following command:
```
python inference_realbasicvsr.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${INPUT_DIR} ${OUTPUT_DIR} --max-seq-len=${MAX_SEQ_LEN} --is_save_as_png=${IS_SAVE_AS_PNG}  --fps=${FPS}
```

This script supports both images and videos as inputs and outputs. You can simply change ${INPUT_DIR} and ${OUTPUT_DIR} to the paths corresponding to the video files, if you want to use videos as inputs and outputs. But note that saving to videos may induce additional compression, which reduces output quality.

For example:
1. Images as inputs and outputs
```
python inference_realbasicvsr.py configs/inference_vsr_x2.py checkpoints/RealBasicVSR_x2.pth data/demo_000 results/demo_000
```

2. Video as input and output
```
python inference_realbasicvsr.py configs/inference_vsr_x2.py checkpoints/RealBasicVSR_x2.pth data/demo_001.mp4 results/demo_001.mp4 --fps=12.5
```


## Training

We only train the second stage from pre-trained weights on REDS dataset. ([Dropbox](https://www.dropbox.com/s/eufigxmmkv5woop/RealBasicVSR.pth?dl=0) / [Google Drive](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view) / [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/chan0899_e_ntu_edu_sg/EfMvf8H6Y45JiY0xsK4Wy-EB0kiGmuUbqKf0qsdoFU3Y-A?e=9p8ITR))

We crop the REDS dataset into sub-images for faster I/O. Please follow the instructions below:
1. Put the original REDS dataset in `./data`
2. Resize the original REDS dataset:
```
python generate_rescale.py
```
3. Run the following command:
```
python crop_sub_images.py --data-root ./data/REDS  --scales 2
```

Then we just train the second stage from pre-trained weights.

```
mim train mmedit configs/vsr_train_st2_128x128.py --gpus 8
```

**Note**: We use UDM10 with bicubic downsampling for validation and use VideoLQ for test. You can also download it from [here](https://www.terabox.com/web/share/link?surl=LMuQCVntRegfZSxn7s3hXw&path=%2Fproject%2Fpfnl) and download the VideoLQ dataset using [Dropbox](https://www.dropbox.com/sh/hc06f1livdhutbo/AAAMPy92EOqVjRN8waT0ie8ja?dl=0) / [Google Drive](https://drive.google.com/drive/folders/1-1iJRNdqdFZWOnoUU4xG1Z1QhwsGwMDy?usp=sharing) / [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/chan0899_e_ntu_edu_sg/ErSugvUBxoBMlvSAHhqT5BEB9-4ZaqxzJIcc9uvVa8JGHg?e=WpHJTc).

## Acknowledgement
[RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR)
