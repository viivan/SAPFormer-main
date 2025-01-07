# SAPFormer
This is the Pytorch implementation of our following paper:



## Environment
This code has been tested with PyTorch  1.10.0  Python  3.8, CUDA 11.3  on Ubuntu 20.04, NVIDIA GeForce 3090 or 2080Ti

1. Create a conda env with
   ```bash
   pip install -r requirement.txt
   ```
2. Compile `pointops`:
   Please make sure the `gcc` and `nvcc` can work normally. Then, compile and install pointops2 by:
    ```bash
    cd lib/pointops2
    python setup.py install
    ```
    (Note that we made the successful compilation under `gcc=7.5.0, cuda=11.3`)
3. Compile `emd` (optional for classification):
    ```bash
    cd lib/emd
    python setup.py install
    ```

## Data preparation
### S3DIS
Please refer to [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch#data-preparation-2) for preprocessing, and put the processed data to `dataset/s3dis/stanford_indoor3d`.

### Scannetv2
Following [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer).

### ShapeNetPart
We follow [PointNext](https://guochengqian.github.io/PointNeXt/examples/shapenetpart/) to uniformly sample 2048 points. You can also use the preprocessed data provided below:


### ModelNet40
Following [PointNext](https://guochengqian.github.io/PointNeXt/examples/shapenetpart/), ModelNet40 dataset will be downloaded automatically.



## Training
The training commands for S3DIS, ShapeNetPart, ModelNet40, and scannetv2 are summarized as follows:
```bash
# S3DIS
./scripts/train_s3dis.sh

# Scannetv2
./scripts/train_scannetv2.sh

# ShapeNetPart
./scripts/train_shapepart.sh

# ModelNet40
./scripts/train_modelnet40.sh 
```

## Testing
 Evaluate on s3dis by simply running:
    

    # using GPU 0
    ./scripts/test_s3dis.sh 0

Other trained models can be similarly evaluated.
    
## Acknowledgment

This repo is built based on [CDFormer](https://github.com/haibo-qiu/CDFormer), [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer), [PointNeXt](https://github.com/guochengqian/PointNeXt). Thanks the contributors of these repos!
