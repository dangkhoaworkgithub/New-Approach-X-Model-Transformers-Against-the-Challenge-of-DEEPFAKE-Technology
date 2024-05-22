# XMT-Model

## New Approach X-Model Transformers Against the Challenge of DEEPFAKE Technology

### Requirements:

- Pytorch >= 1.4

### Preprocessing:

Face extraction from video. 

We use pretrained YoloV5 on the face for more accurate face recognition.

pip install -r `requirements.txt`


### Train:
pip install -r `requirements.txt`

To train the model on your own, you can use the following parameters:

- `e`: epoch
- `s`: session - (`g`) - GPU or (`c`) - CPU
- `w`: weight decay (default: 0.0000001)
- `l`: learning rate (default: 0.001)
- `d`: path file
- `b`: batch size (default: 32)
- `p`: The process of accuracy and loss

#### Example command:

To train the model using specific parameters:

```bash
python train.py -e 15 -s 'g' -l 0.0001 -w 0.0000001 -d data/sample_train_data/ -p
```
### Weights:
`xmodel_deepfake_sample.pth`: Full weight for detection.
### Predict XMT:
- Predict on Image
```bash
python image-prediction.py
```
- Predict on Video

```bash
python video-prediction.py --video_path --display
```
```bash
python video-prediction.py --video_path --output_path --save
```
### Authors:
- Le Dang Khoa
- Le Gia Hung
- Nguyen Hung Thinh

