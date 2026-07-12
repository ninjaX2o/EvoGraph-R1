# [CVPR 2026] EvoGraph-R1: Self-Evolving Multimodal Knowledge Hypergraphs for Agentic Retrieval

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_EvoGraph-R1_Self-Evolving_Multimodal_Knowledge_Hypergraphs_for_Agentic_Retrieval_CVPR_2026_paper.html)
[![Paper](https://img.shields.io/badge/Paper-CVF-red)](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_EvoGraph-R1_Self-Evolving_Multimodal_Knowledge_Hypergraphs_for_Agentic_Retrieval_CVPR_2026_paper.pdf)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green)](#get-started)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Apache--2.0-lightgrey)](LICENSE)

Official implementation of **EvoGraph-R1: Self-Evolving Multimodal Knowledge
Hypergraphs for Agentic Retrieval**. A GraphRAG framework that models retrieval as an MDP over a dynamic hypergraph, enabling an agent to query, expand, edit, and answer through closed-loop graph evolution.

## Get Started

Create the environment:

```bash
conda create -n evograph-r1 python=3.11
conda activate evograph-r1

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
pip install -e . --no-deps
```

Set API keys:

```bash
cp .env.example .env

export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=gpt-4o-mini
export JINA_API_KEY=...
```

## Data

Raw datasets and generated graph artifacts are not included. Keep local files
under ignored directories:

```text
datasets/      Text QA data
datasets_mm/   Multimodal VQA, KB, and image data
expr/          Text hypergraphs and retrieval indexes
expr_mm/       Multimodal graph and index artifacts
```

Text datasets used in the paper:

| Dataset | Source |
| --- | --- |
| 2WikiMultiHopQA | [Official repository](https://github.com/Alab-NII/2wikimultihop) |
| HotpotQA | [Official website](https://hotpotqa.github.io/) |
| Natural Questions | [Official website](https://ai.google.com/research/NaturalQuestions) |

Multimodal datasets and assets:

| Resource | Source |
| --- | --- |
| EchoSight assets | [EchoSight repository](https://github.com/Go2Heart/EchoSight) |
| Encyclopedic VQA / E-VQA | [Google Research release](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa) |
| InfoSeek | [Project page](https://open-vision-language.github.io/infoseek/) |
| OK-VQA | [Dataset page](https://okvqa.allenai.org/) |
| OVEN images | [Hugging Face](https://huggingface.co/datasets/ychenNLP/oven) |
| Google Landmarks v2 | [Dataset page](https://github.com/cvdfoundation/google-landmark) |
| iNaturalist 2021 | [Dataset page](https://github.com/visipedia/inat_comp/tree/master/2021) |

## Text EvoGraph-R1 Construction

Prepare data in the text EvoGraph-R1 layout:

```text
datasets/<DATA_SOURCE>/raw/qa_train.json
datasets/<DATA_SOURCE>/raw/qa_dev.json
datasets/<DATA_SOURCE>/raw/qa_test.json
```

Preprocess and build the knowledge hypergraph:

```bash
export DATA_SOURCE=<text-dataset-name>

python script_process.py --data_source ${DATA_SOURCE}
python script_build.py --data_source ${DATA_SOURCE}
```

## Multimodal Graph Construction

For EchoSight-style E-VQA and InfoSeek assets, generate a local placement plan:

```bash
export MM_DATA_ROOT=/path/to/evograph-mm-data
export MM_DATASET=<E-VQA-or-InfoSeek>
export MM_SUBSET=<subset-name>

python script_prepare_echosight_mm.py --root ${MM_DATA_ROOT} --dataset ${MM_DATASET} --write-plans --required-only
```

After placing assets, validate, preprocess, and build the multimodal graph:

```bash
python script_validate_echosight_mm.py --root ${MM_DATA_ROOT} --dataset ${MM_DATASET}
python script_process_mm.py --root ${MM_DATA_ROOT} --dataset ${MM_DATASET} --subset ${MM_SUBSET} --metadata-only --format parquet
python script_build_mm.py --root ${MM_DATA_ROOT} --dataset ${MM_DATASET} --subset ${MM_SUBSET} --output-root ${MM_DATA_ROOT}/expr_mm --embedding-backend ${MM_EMBEDDING_BACKEND} --model ${MM_EMBED_MODEL_PATH}
```

For OK-VQA or custom multimodal datasets, prepare the same processed parquet
and graph/index layout under `${MM_DATA_ROOT}/datasets_mm/${MM_DATASET}/` and
`${MM_DATA_ROOT}/expr_mm/${MM_DATASET}/`.

## Training

### Text EvoGraph-R1

Start the retrieval API and run GRPO:

```bash
export DATA_SOURCE=<text-dataset-name>

python script_api.py --data_source ${DATA_SOURCE} --port 8001
bash run_grpo.sh -p Qwen/Qwen2.5-7B-Instruct -m Qwen2.5-7B-Instruct -d ${DATA_SOURCE}
```

Other text RL launchers:

```bash
bash run_rpp.sh -p Qwen/Qwen2.5-7B-Instruct -m Qwen2.5-7B-Instruct -d ${DATA_SOURCE}
bash run_ppo.sh -p Qwen/Qwen2.5-7B-Instruct -m Qwen2.5-7B-Instruct -d ${DATA_SOURCE}
```

### Multimodal EvoGraph-R1

Start the text and multimodal APIs:

```bash
export TEXT_DATA_SOURCE=<text-dataset-name>
export MM_DATASET=<multimodal-dataset-name>
export MM_SUBSET=<subset-name>
export MM_KB_DIR=${MM_DATA_ROOT}/expr_mm/${MM_DATASET}
export MM_EMBED_DEVICE=cpu

python script_api.py --data_source ${TEXT_DATA_SOURCE} --working_dir expr/${TEXT_DATA_SOURCE} --port 8001
python script_api_mm.py --dataset ${MM_DATASET} --subset ${MM_SUBSET} --working_dir ${MM_KB_DIR} --embedding-backend ${MM_EMBEDDING_BACKEND} --model ${MM_EMBED_MODEL_PATH} --port 8003
```

Run multimodal GRPO:

```bash
export ACTOR_LR=5e-7

bash run_mm_grpo.sh \
  -p Qwen/Qwen2.5-VL-7B-Instruct \
  -m Qwen2.5-VL-7B-Instruct \
  -d ${MM_DATASET} \
  -s ${MM_SUBSET}
```

The 3B variants can be used as lower-resource alternatives by replacing the
model path and name.

Use environment variables such as `ACTOR_LR`, `N_GPUS`, `TRAIN_BATCH_SIZE`,
`VAL_BATCH_SIZE`, and `TOTAL_EPOCHS` to tune and scale training.

## Citation

```bibtex
@inproceedings{lin2026evograph,
  title={EvoGraph-R1: Self-Evolving Multimodal Knowledge Hypergraphs for Agentic Retrieval},
  author={Lin, Jiashi and Jiang, Changhong and Lin, Xiangru and Zhang, Ruifei and Zhu, Xinyi and Liu, Jiyao and Tang, Cheng and Du, Ye and Gao, Shujian and Ning, Junzhi and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={756--765},
  year={2026}
}
```

## Acknowledgements

This repository builds on [Graph-R1](https://github.com/LHRLAB/Graph-R1),
[EchoSight](https://github.com/Go2Heart/EchoSight),
[Search-R1](https://github.com/PeterGriffinJin/Search-R1), and
[verl](https://github.com/verl-project/verl). We thank the authors for
open-sourcing their code and resources.
