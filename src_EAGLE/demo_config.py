import os
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join
from modules import *
import hydra
import torch.multiprocessing
from PIL import Image
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from train_segmentation_eigen import LitUnsupervisedSegmenter
from tqdm import tqdm
import random
from data import create_pascal_label_colormap, ContrastiveSegDataset

torch.multiprocessing.set_sharing_strategy('file_system')


def create_cocostuff_colormap():
    colors = np.array([
        [0, 192, 64],
        [0, 192, 64],
        [0, 64, 96],
        [128, 192, 192],
        [0, 64, 64],
        [0, 192, 224],
        [0, 192, 192],
        [128, 192, 64],
        [0, 192, 96],
        [128, 192, 64],
        [64, 192, 64],
        [64, 64, 0],
        [0, 128, 192],
        [0, 192, 32],
        [64, 32, 32],
        [0, 192, 0],
        [0, 0, 0],
        [192, 192, 160],
        [0, 32, 0],
        [0, 192, 32],
        [0, 128, 160],
        [192, 64, 64],
        [0, 192, 160],
        [192, 192, 0],
        [0, 64, 32],
        [192, 32, 192],
        [192, 192, 96],
    ], dtype=np.uint8)
    padded = np.zeros((512, 3), dtype=np.uint8)
    padded[:len(colors)] = colors
    return padded


class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = join(root)
        self.transform = transform
        self.images = os.listdir(self.root)

    def __getitem__(self, index):
        image = Image.open(join(self.root, self.images[index])).convert('RGB')
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        return image, self.images[index]

    def __len__(self):
        return len(self.images)


def get_cluster_assignments(model, cfg):
    """Run validation set through model to compute cluster assignments"""
    print("Computing cluster assignments from validation set...")
    
    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=cfg.pytorch_data_dir,
        dataset_name=model.cfg.dataset_name,
        crop_type=None,
        img_set="val",
        transform=get_transform(model.cfg.res, False, "center"),
        target_transform=get_transform(model.cfg.res, True, "center"),
        mask=True,
        cfg=model.cfg,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4,
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    model.eval().cuda()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Computing assignments"):
            img = batch["img"].cuda()
            label = batch["label"].cuda()
            
            feats, feats_kk, code, code_kk = model.net(img)
            code_kk = F.interpolate(code_kk, label.shape[-2:], 
                                     mode='bilinear', align_corners=False)
            
            # Update cluster metrics to compute assignments
            _, cluster_preds = model.cluster_probe(code_kk, None)
            cluster_preds = cluster_preds.argmax(1)
            model.cluster_metrics.update(cluster_preds, label)
    
    # This computes the assignments
    model.cluster_metrics.compute(training=False)
    print("Cluster assignments computed successfully!")


@hydra.main(config_path="configs", config_name="demo_config.yaml")
def my_app(cfg: DictConfig) -> None:
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "linear"), exist_ok=True)

    # Load model
    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.model_path)
    print(OmegaConf.to_yaml(model.cfg))

    # Compute cluster assignments first
    get_cluster_assignments(model, cfg)

    # Use cocostuff colormap
    label_cmap = create_cocostuff_colormap()

    dataset = UnlabeledImageFolder(
        root=cfg.image_dir,
        transform=get_transform(cfg.res, False, "center"),
    )

    loader = DataLoader(
        dataset,
        cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate
    )

    model.eval().cuda()
    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net

    for i, (img, name) in enumerate(tqdm(loader)):
        with torch.no_grad():
            img = img.cuda()

            # Forward pass with test time augmentation
            out1 = par_model(img)
            out2 = par_model(img.flip(dims=[3]))

            # net returns: feats, feats_kk, code, code_kk
            code1 = out1[3]
            code2 = out2[3]

            # Average predictions
            code = (code1 + code2.flip(dims=[3])) / 2

            # Upsample to original image size
            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            # Get linear probe predictions
            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()

            # Get cluster probe predictions
            _, cluster_probs = model.cluster_probe(code, 2, log_probs=True)
            cluster_probs = cluster_probs.cpu()

            for j in range(img.shape[0]):
                single_img = img[j].cpu()

                # Apply dense CRF
                linear_crf = dense_crf(single_img, linear_probs[j]).argmax(0)
                cluster_crf = dense_crf(single_img, cluster_probs[j]).argmax(0)

                # Apply colormap to linear predictions
                linear_colored = label_cmap[linear_crf].astype(np.uint8)

                # Map clusters then apply colormap
                cluster_mapped = model.cluster_metrics.map_clusters(cluster_crf)
                cluster_colored = label_cmap[cluster_mapped].astype(np.uint8)

                # Save results
                new_name = ".".join(name[j].split(".")[:-1]) + ".png"
                Image.fromarray(linear_colored).save(join(result_dir, "linear", new_name))
                Image.fromarray(cluster_colored).save(join(result_dir, "cluster", new_name))

                # Save original image for comparison
                orig_img = img[j].cpu().permute(1, 2, 0).numpy()
                orig_img = (orig_img * np.array([0.229, 0.224, 0.225]) +
                            np.array([0.485, 0.456, 0.406])) * 255
                orig_img = orig_img.clip(0, 255).astype(np.uint8)
                Image.fromarray(orig_img).save(join(result_dir, "orig_" + new_name))


if __name__ == "__main__":
    prep_args()
    my_app()