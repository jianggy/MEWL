from dataset import LexiconCLIPData
from model.clip_trans import CLIPTrans, CLIPTransWOTextEncoder
import pytorch_lightning as pl
from pytorch_lightning import Trainer

if __name__ == "__main__":
    module = CLIPTransWOTextEncoder(clip_model='/scratch/lmcp/pllm/ViT-B-16.pt', d_model=512,
                                    num_layers=6, nhead=8, dim_feedforward=2048, n_classes=5, dropout=0.1, context_length=20)

    trainer = Trainer(accelerator="gpu", devices=4, num_nodes=1, max_epochs=600,
                      gradient_clip_val=1.0, check_val_every_n_epoch=3, log_every_n_steps=5)

    trainer.fit(module, datamodule=LexiconCLIPData(
        dataset_path='/scratch/lmcp/lexicon/lexiconv5', batch_size=64, context_length=20))

    trainer.test(module, datamodule=LexiconCLIPData(
        dataset_path='/scratch/lmcp/lexicon/lexiconv5', batch_size=64, context_length=20))