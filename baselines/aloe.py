from dataset import MEWLCLIPData, MEWLAloeData
# from model.clip_trans import CLIPTrans, CLIPTransWOTextEncoder
from model.aloe import Aloe
import pytorch_lightning as pl
from pytorch_lightning import Trainer

if __name__ == "__main__":
    module = Aloe(d_model=512, num_layers=16, nhead=8, dim_feedforward=1024, head_size=144, n_classes=5, dropout=0.1, context_length=20, seed=0)

    trainer = Trainer(accelerator="gpu", devices=4, num_nodes=1, max_epochs=600,
                      gradient_clip_val=1.0, check_val_every_n_epoch=3, log_every_n_steps=5)

    trainer.fit(module, datamodule=MEWLAloeData(
        dataset_path='/home/xinshiji/project/conceptaloe/MEWLv6_noimg', batch_size=64, context_length=20))

    trainer.test(module, datamodule=MEWLCLIPData(
        dataset_path='/scratch/lmcp/MEWL/MEWLv5', batch_size=64, context_length=20))