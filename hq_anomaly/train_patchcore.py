import torch
import torch.distributed
import torchvision.datasets 
import torchvision.transforms
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))
from hq_anomaly import common
from hq_anomaly import models
import sys
from tqdm import tqdm
import numpy as np
from hq_anomaly import valid_patchcore
from hq_anomaly.datasets import ImageSingleFolder


def create_model(config: common.ModelConfig) -> torch.nn.Module:
    # model = models.AutoEncoderViT()
    model = models.ViTPatchcore(model_config=config)
    return model

def train(config: common.TrainConfig):
    image_size = config.modelConfig.image_size
    data_path = config.data_path

    print("Not using distributed training")
    device = torch.device(f"cuda:{config.devices[0]}" if torch.cuda.is_available() else "cpu")
    rank = 0

    # data loader
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.GaussianBlur(kernel_size=3, sigma=1.0),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])
    # train_path = os.path.join(data_path, 'train', 'good')
    # valid_path = os.path.join(data_path, 'test')
    train_path = os.path.join(data_path, 'train')
    valid_path = os.path.join(data_path, 'val')
    
    train_dataset = ImageSingleFolder(
        folder=train_path, transform=transforms)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_data_workers,
        sampler=torch.utils.data.DistributedSampler(train_dataset) if torch.distributed.is_initialized() else torch.utils.data.RandomSampler(train_dataset),
    )

    model = create_model(config.modelConfig)
    model.to(device)
    model.train()

    os.makedirs(config.output_path, exist_ok=True)
    for i_epoch in range(len(model.layer_indices)):
        if rank == 0:
            bar = tqdm(train_loader)
        else:
            bar = train_loader
            pass

        # model.set_require_grad_(i_epoch)
        for i_batch, images in enumerate(bar):
            images = images.to(device)
            forward_result = model(images)
            model.compute_loss(forward_result, i_epoch)

            # if i_batch > 10:
            #     break
            pass
        model.shrink_memory(i_epoch)
        pass
    # model.load("output/ckpt.pth")
    
    dist_stats, confidence, accuracy, f1_score, precision, recall, precision_recall_curve, roc_curve, fpr_zero_miss = valid_patchcore.valid(model, folder=valid_path)
    model.set_distance_stats(dist_stats)

    model.save(os.path.join(config.output_path, "ckpt.pth"))
    print(f"middle_dist: {dist_stats}, accuracy: {accuracy}, f1_score: {f1_score}, precision: {precision}, recall: {recall}")
    results_filename = os.path.join(config.output_path, "results.csv")

    with open(results_filename, 'w') as fout:
        # write header
        fout.write("confidence,accuracy,f1_score,precision,recall,fpr_zero_miss\n")
        fout.write(f"{confidence},{accuracy},{f1_score},{precision},{recall},{fpr_zero_miss}\n")
        pass
    precision_recall_curve_filename = os.path.join(config.output_path, "pr_curve_metric.csv")
    with open(precision_recall_curve_filename, 'w') as fout:
        # write header
        fout.write("px,all\n")
        for p, r in zip(precision_recall_curve[1], precision_recall_curve[0]):
            fout.write(f"{p:.4f},{r:.4f}\n")
            pass
        pass
    roc_curve_filename = os.path.join(config.output_path, "roc_curve_metric.csv")
    with open(roc_curve_filename, 'w') as fout:
        # write header
        fout.write("fpr,tpr,thr\n")
        for f, t, h in zip(roc_curve[0], roc_curve[1], roc_curve[2]):
            fout.write(f"{f:.4f},{t:.4f},{h:.4f}\n")
            pass
        pass



if __name__ == "__main__":
    train_config = common.TrainConfig(
        data_path="/root/autodl-tmp/xianshuban_0721",
        output_path="/root/hq_anomaly/hq_anomaly/output/0721_xianshuban",
        batch_size=1,
        num_epochs=200,
        num_data_workers=16,
        modelConfig=common.ModelConfig(
            image_size=512,
        ),
    )
    train(train_config)
    pass