# YOLO + Deep Sort 

pip install -r requirements.txt

 Download YOLOv3 parameters
```
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
cd ../../../
```

Download deepsort parameters ckpt.t7
```
cd deep_sort/deep/checkpoint
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
cd ../../../
```  

Compile nms module
```bash
cd detector/YOLOv3/nms
sh build.sh
cd ../../..
```

Notice:
If compiling failed, the simplist way is to **Upgrade your pytorch >= 1.1 and torchvision >= 0.3" and you can avoid the troublesome compiling problems which are most likely caused by either `gcc version too low` or `libraries missing`.

Run demo
```
usage: python yolov3_deepsort.py VIDEO_PATH
                                [--help]
                                [--frame_interval FRAME_INTERVAL]
                                [--config_detection CONFIG_DETECTION]
                                [--config_deepsort CONFIG_DEEPSORT]
                                [--display]
                                [--display_width DISPLAY_WIDTH]
                                [--display_height DISPLAY_HEIGHT]
                                [--save_path SAVE_PATH]          
                                [--cpu]          

# yolov3 + deepsort
python yolov3_deepsort.py [VIDEO_PATH]

# yolov3_tiny + deepsort
python yolov3_deepsort.py [VIDEO_PATH] --config_detection ./configs/yolov3_tiny.yaml

# yolov3 + deepsort on webcam
python3 yolov3_deepsort.py /dev/video0 --camera 0

# yolov3_tiny + deepsort on webcam
python3 yolov3_deepsort.py /dev/video0 --config_detection ./configs/yolov3_tiny.yaml --camera 0
```
Use `--display` to enable display.  
Results will be saved to `./output/results.avi` and `./output/results.txt`.


## Training the RE-ID model
dataset Market1501 Preparation 参考Person_ReID

把生成的pytorch文件夹copy到~/yolo_deepsort_pytorch/deep_sort/deep并重命名为data_market1501

dataset Mars Preparation  https://www.kaggle.com/twoboysandhats/mars-motion-analysis-and-reidentification-set

下载Mars数据集 并重命名为data_mars 修改子文件名称改为train和test

Then you can try [train.py](deep_sort/deep/train.py) to train your own parameter and evaluate it using [test.py](deep_sort/deep/test.py) and [evaluate.py](deep_sort/deep/evalute.py).


## References
https://github.com/ZQPei/deep_sort_pytorch
