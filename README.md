# EAGLE :eagle:: Eigen Aggregation Learning for Object-Centric Unsupervised Semantic Segmentation
### Abstract
Semantic segmentation has innately relied on extensive pixel-level annotated data, leading to the emergence of unsupervised methodologies. Among them, leveraging self-supervised Vision Transformers for unsupervised semantic segmentation (USS) has been making steady progress with expressive deep features. Yet, for semantically segmenting images with complex objects, a predominant challenge remains: the lack of explicit object-level semantic encoding in patch-level features. This technical limitation often leads to inadequate segmentation of complex objects with diverse structures. To address this gap, we present a novel approach, **EAGLE**, which emphasizes object-centric representation learning for unsupervised semantic segmentation. Specifically, we introduce EiCue, a spectral technique providing semantic and structural cues through an eigenbasis derived from the semantic similarity matrix of deep image features and color affinity from an image. Further, by incorporating our object-centric contrastive loss with EiCue, we guide our model to learn object-level representations with intra- and inter-image object-feature consistency, thereby enhancing semantic accuracy. Extensive experiments on COCO-Stuff, Cityscapes, and Potsdam-3 datasets demonstrate the state-of-the-art USS results of EAGLE with accurate and consistent semantic segmentation across complex scenes.

## :book: Contents
<!--ts-->
   * [Installation](#installation)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [About EAGLE](#about-eagle)
      * [Eigenvectors](#eigenvectors)
      * [EiCue](#eicue)
      * [EAGLE Architecture](#eagle-architecture)
      * [Results](#results)
   * [Citation](#citation)

<!--te-->


## Installation


### Install Conda Environment
```shell script
conda env create -f environment.yml
conda activate EAGLE
```
We share the same Anaconda environment with [STEGO](https://github.com/mhamilton723/STEGO/tree/master).

### Download Pre-Trained Models
```shell script
cd src_EAGLE
python download_models.py
```

### Download Datasets
Modify the `pytorch_data_dir` to your own data directory and run download_datasets.py. 
Then, go to your data directory and extract the files from the zip archive:

```shell script
python download_datasets.py

cd /YOUR/DATA/DIR
unzip cocostuff.zip
unzip cityscapes.zip
unzip potsdam.zip
unzip potsdamraw.zip
```

## 🦾Training
First, create a cropped dataset, and consider altering the `crop_datasets.py` script to select the specific dataset you want to crop.

```shell script
python crop_datasets.py
```

Next, run the following in `EAGLE/src_EAGLE`:
```shell script
python train_segmentation_eigen.py
```

You can modify hyperparameters in the file located at [`EAGLE/src_EAGLE/configs/train_config_cocostuff.yml`](src_EAGLE/configs/train_config_cocostuff.yml)

## 🦿Evaluation
To evaluate please run the following in `EAGLE/src`:
```shell script
python eval_segmentation.py
```
You can modify evaluation hyperparameters in the file located at [`EAGLE/src_EAGLE/configs/eval_config.yml`](src_EAGLE/configs/eval_config.yml)

### Checkpoints
We release the weights on trained EAGLE:

<table style="margin: auto">
  <tr>
    <th>Dataset</th>
    <th>backbone</th>
    <th>checkpoint drive</th>
  </tr>
  <tr>
    <td align="center">COCO-Stuff</td>
    <td align="center">ViT-S/8</td>
    <td><a href="https://drive.google.com/file/d/1fRZB_Tx2cZn5XayY0MiC9gv6D9kML7lh/view?usp=sharing">link</td>
  </tr>
  <tr>
    <td align="center">Cityscapes</td>
    <td align="center">ViT-B/8</td>
    <td><a href="https://drive.google.com/file/d/1W943QkhcnD2l3Ye58ovG5fsGdZ1uFpuH/view?usp=sharing">link</td>
  </tr>
</table>

## :eagle: About EAGLE

### Eigenvectors
Visualizing eigenvectors derived from S in the Eigen Aggregation Module. These eigenvectors not only distinguish different objects but also identify semantically related areas, highlighting how EiCue captures object semantics and boundaries effectively.
![eigenvector](./img/eigenvector.png)

### EiCue
Comparison between K-means and EiCue. The bottom row presents EiCue, highlighting its superior ability to capture subtle structural intricacies and understand deeper semantic relationships, which is not as effectively achieved by K-means.
![EiCue](./img/eicue.png)

### EAGLE Architecture
The pipeline of **EAGLE**. Leveraging the Laplacian matrix, which integrates hierarchically projected image key features and color affinity, the model harnesses eigenvector clustering to capture an object-centric perspective. Our model further adopts an object-level contrastive loss, utilizing the projected vector Z and ̃Z. The learnable prototype Φ, acts as a singular anchor that contrasts positive objects and negative objects. Our object-level contrastive loss is computed in two distinct manners: crosswise and non-crosswise to ensure semantic consistency.

![image](./img/mainfigure.png)


### Results
We evaluate the EAGLE algorithm on the CocoStuff, Cityscapes, and Potsdam-3 datasets.

![main_results](./img/results.png)


