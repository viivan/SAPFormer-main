# SAPFormer

This is the Pytorch implementation of our following paper:

## Environment

1. Create a conda env with
   ```bash
   conda env create -f environment.yml
   ```
2. Compile `pointops`:
   Please make sure the `gcc` and `nvcc` can work normally. Then, compile and install pointops2 by:
   ```bash
   cd lib/pointops2
   python setup.py install
   ```
   (Note that we made the successful compilation under `gcc=7.5.0, cuda=11.3`)

## Data preparation

### S3DIS

Please refer to [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch#data-preparation-2) for preprocessing, and put the processed data to `dataset/s3dis/stanford_indoor3d`.

Other datasets are coming soon
## Training

The training commands are summarized as follows:

```bash
# S3DIS
python  train_seg.py --config config/s3dis/s3dis_sapformer.yaml debug 0


## Testing
python  test_seg.py --config config/s3dis/s3dis_sapformer.yaml model_path 'your_path' save_folder 'temp/s3dis/results'

## Acknowledgment

This repo is built based on  [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer), [PointNeXt](https://github.com/guochengqian/PointNeXt). Thanks the contributors of these repos!

```
We extend our thanks to Ge Sihan for the implementation of the code for this algorithm.
