# <img src="./figures/cat.png" height="20"/>MEWL

This repo contains code for our ICML 2023 paper:

[<img src="./figures/cat.png" height="15"/>MEWL: Few-shot multimodal word learning with referential uncertainty](https://arxiv.org/abs/2306.00503)

[Guangyuan Jiang](https://jianggy.com), [Manjie Xu](https://mjtsu.github.io), [Shiji Xin](https://www.shijixin.net), [Wei Liang](https://liangwei-bit.github.io/web/), [Yujia Peng](https://yujiapeng.com), [Chi Zhang](http://wellyzhang.github.io), and [Yixin Zhu](https://yzhu.io)

ICML 2023

Without explicit feedback, humans can rapidly learn the meaning of words. Children can acquire a new word after just a few passive exposures, a process known as fast mapping. This word learning capability is believed to be the most fundamental building block of multimodal understanding and reasoning. Despite recent advancements in multimodal learning, a systematic and rigorous evaluation is still missing for human-like word learning in machines. To fill in this gap, we introduce the MachinE Word Learning (<img src="./figures/cat.png" height="15"/>MEWL) benchmark to assess how machines learn word meaning in grounded visual scenes. <img src="./figures/cat.png" height="15"/>MEWL covers human's core cognitive toolkits in word learning: cross-situational reasoning, bootstrapping, and pragmatic learning. Specifically, <img src="./figures/cat.png" height="15"/>MEWL is a few-shot benchmark suite consisting of nine tasks for probing various word learning capabilities. These tasks are carefully designed to be aligned with the children's core abilities in word learning and echo the theories in the developmental literature. By evaluating multimodal and unimodal agents' performance with a comparative analysis of human performance, we notice a sharp divergence in human and machine word learning. We further discuss these differences between humans and machines and call for human-like few-shot word learning in machines.

Dataset link: https://doi.org/10.5281/zenodo.7993374



<img src="./figures/example.png" />



## Dependencies

### Dataset rendering

- [Blender 3.3 LTS](https://www.blender.org/download/releases/3-3/)
- Python 3.10 (Blender)

### Baselines

We recommend using conda: 

```bash
conda create --name mewl --file environment.yml
```



## Usage

### Dataset download

You can download the <img src="./figures/cat.png" height="15"/>MEWL dataset [here from Zenodo](https://doi.org/10.5281/zenodo.7993374).


### Dataset generation

#### Step 1 (Clone)

Clone this repo.

```bash
git clone git@github.com:jianggy/MEWL.git
```

Init submodule: modifed [CLEVR generation code](https://github.com/jianggy/clevr-dataset-gen) that supports Blender 3.0+.

```bash
git submodule update --init --recursive
```

#### Step 2 (CLEVR generation code setup)

Install Blender 3.3 LTS from [here](https://www.blender.org/download/releases/3-3/).

Blender ships with its own installation of Python which is used to execute scripts that interact with Blender; you'll need to add the `image_generation` directory to Python path of Blender's bundled Python. The easiest way to do this is by adding a `.pth` file to the `site-packages` directory of Blender's Python, like this:

```bash
echo $PWD/clevr-dataset-gen/image_generation >> $BLENDER/3.3/python/lib/python3.10/site-packages/clevr.pth
```

where `$BLENDER` is the directory where Blender is installed and `$VERSION` is your Blender version; for example on OSX you might run:

```bash
echo $PWD/clevr-dataset-gen/image_generation >> /Applications/blender/blender.app/Contents/Resources/3.3/python/lib/python3.10/site-packages/clevr.pth
```

On OSX the `blender` binary is located inside the blender.app directory; for convenience you may want to add the following alias to your `~/.bash_profile` file:

```bash
alias blender='/Applications/blender/blender.app/Contents/MacOS/blender'
```

#### Step 3 (Generate MEWL)

```bash
$BLENDER/blender --background --python generate_mewl.py -- --type <shape/color/material/object/composite/relation/bootstrap/number/pragmatic> --save_path <path to save rendered images> --log_path <path to save generation log> --start_idx <start episode id> --end_idx <end episode id>
```

To enable multi-process rendering, you can make ``--start_idx`` and ``--end_idx`` non-interleaved intervals.

### Dataset structure

After unzipping the downloaded file, you will see a file structure like this:

```
MEWL_Release_V1/
├─ test/
│  ├─ bootstrap/
│  │  ├─ 0/
│  │  │  ├─ info.json
│  │  │  ├─ question.json
│  │  │  ├─ question.png
│  │  │  ├─ selardy front senprever.json
│  │  │  ├─ selardy front senprever.png
│  │  │  ├─ selardy right pecibe.json
│  │  │  ├─ selardy right pecibe.png
│  │  │  ├─ tuevelar behind senprever.json
│  │  │  ├─ tuevelar behind senprever.png
│  │  │  ├─ upexments front selardy.json
│  │  │  ├─ upexments front selardy.png
│  │  │  ├─ upexments left senprever.json
│  │  │  ├─ upexments left senprever.png
│  │  │  ├─ upexments left tuevelar.json
│  │  │  ├─ upexments left tuevelar.png
│  │  ├─ 1/
│  ├─ color/
│  ├─ composite/
│  ├─ material/
│  ├─ number/
│  ├─ object/
│  ├─ pragmatic/
│  ├─ relation/
│  ├─ shape/
├─ train/
├─ val/

```

In the released version of <img src="./figures/cat.png" height="15"/>MEWL, we organize train, test, and validation split into three folders `train/`, `test/`, and `val/`. Inside each split, you will see the nine task folders.

Each subfolder contains images and meta info. In the example episode shown above (`0/`), `selardy front senprever.png`, `selardy right pecibe.png`, `tuevelar behind senprever.png`, `upexments front selardy.png`, `upexments left senprever.png`, `upexments left tuevelar.png` are the six context panel images, and the filenames are the corresponding utterances.

`question.png` is the query image. The five candidate options and the ground choice answer are in the `question['choices']` and `question['answer']` fields of the `info.json` file. You can also find additional metadata in the corresponding JSON files, for example, scene descriptions, ground-truth `word-concept` mapping, and `3d_coords` for converting to object masks (see the NS-VQA paper for this operation).



## Baselines

All the baseline codes are in the `baselines/` folder.

#### CLIP

Run on four NVIDIA A100 80GB GPUs with slurm:

```bash
srun python clip_trans.py
```

#### Aloe

Run on four NVIDIA A100 80GB GPUs with slurm:

```bash
srun python aloe.py
```

#### BERT

Run `huggingface_preprocess.ipynb` to get a unimodal, captioned version of MEWL.

Run on four NVIDIA A100 80GB GPUs with slurm:

```bash
torchrun --nnodes 1 --nproc_per_node 4 \
bert.py \
--model_name_or_path bert-base-uncased \
--do_eval \
--seed 42 \
--per_device_train_batch_size 8 \
--report_to tensorboard \
--per_device_eval_batch_size 8 \
--save_steps 20000 \
--save_total_limit 20 \
--learning_rate 5e-5 \
--num_train_epochs 200 \
--evaluation_strategy epoch \
--logging_steps 3 \
--output_dir ./log \
--overwrite_output \
--train_file <path to unimodal MEWL> \
--do_train
```

#### Flamingo

Download the pre-trained Flamingo mini from [here](https://github.com/dhansmair/flamingo-mini).

Run on eight NVIDIA A100 80GB GPUs with slurm:

```bash
torchrun --nnodes 1 --nproc_per_node 8 \
flamingo.py \
--model_name <path to pretrained Flamingo checkpoint> \
--do_eval \
--seed 42 \
--per_device_train_batch_size 12 \
--report_to tensorboard \
--per_device_eval_batch_size 12 \
--save_steps 10000 \
--save_total_limit 20 \
--learning_rate 5e-5 \
--num_train_epochs 200 \
--evaluation_strategy epoch \
--logging_steps 3 \
--output_dir ./log \
--overwrite_output \
--dataset_path <path to MEWL> \
--do_train \
--dataloader_num_workers 16
```



## Citation

If you find the paper and/or the code helpful, please cite us.

```bibtex
@inproceedings{jiang2023mewl,
  title={MEWL: Few-shot multimodal word learning with referential uncertainty},
  author={Jiang, Guangyuan and Xu, Manjie and Xin, Shiji and Liang, Wei and Peng, Yujia and Zhang, Chi and Zhu, Yixin},
  booktitle={ICML},
  year={2023}
}
```
