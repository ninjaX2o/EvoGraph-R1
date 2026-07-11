import os
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
VERSION = os.getenv("EVOGRAPH_R1_VERSION", "0.1.0")

install_requires = [
    "accelerate",
    "aioboto3",
    "aiohttp",
    "beautifulsoup4",
    "codetiming",
    "datasets",
    "dill",
    "faiss-cpu",
    "fastapi",
    "filelock",
    "FlagEmbedding",
    "googlesearch-python",
    "graspologic",
    "httpx",
    "huggingface-hub",
    "hydra-core",
    "ijson",
    "jsonlines",
    "math-verify[antlr4_9_3]",
    "nano-vectordb",
    "networkx",
    "nltk",
    "numpy",
    "omegaconf",
    "ollama",
    "openai>=1.40.0",
    "packaging",
    "pandas",
    "peft",
    "pillow",
    "pyarrow>=15.0.0",
    "pybind11",
    "pydantic",
    "pylatexenc",
    "python-dotenv",
    "ray[default]>=2.10",
    "requests",
    "safetensors",
    "sentence-transformers",
    "spacy",
    "sympy",
    "tenacity",
    "tensordict<0.6",
    "tiktoken",
    "torchdata",
    "tqdm",
    "transformers",
    "typing_extensions",
    "uvicorn",
    "vllm==0.7.3",
    "wandb",
    "xxhash",
]

TEST_REQUIRES = ["pytest", "yapf", "py-spy"]
PRIME_REQUIRES = ["pyext"]
GEO_REQUIRES = ["mathruler"]
GPU_REQUIRES = ["liger-kernel", "flash-attn==2.6.3"]
DB_REQUIRES = [
    "hnswlib",
    "neo4j",
    "oracledb",
    "pymilvus",
    "pymongo",
    "pymysql",
    "pyvis",
    "sqlalchemy",
]
TRACKING_REQUIRES = ["mlflow", "swanlab"]
MEGATRON_REQUIRES = ["apex", "cupy", "einops"]
LMDEPLOY_REQUIRES = ["lmdeploy"]

extras_require = {
    "test": TEST_REQUIRES,
    "prime": PRIME_REQUIRES,
    "geo": GEO_REQUIRES,
    "gpu": GPU_REQUIRES,
    "db": DB_REQUIRES,
    "tracking": TRACKING_REQUIRES,
    "megatron": MEGATRON_REQUIRES,
    "lmdeploy": LMDEPLOY_REQUIRES,
}

long_description = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="evograph-r1",
    version=VERSION,
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.11",
    license="MIT AND Apache-2.0",
    license_files=["LICENSE", "LICENSES/Apache-2.0.txt"],
    author="EvoGraph-R1 contributors",
    description="RL training and retrieval stack for editable text and multimodal graph reasoning",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml", "trainer/runtime_env.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
