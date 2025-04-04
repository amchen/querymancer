import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ModelProvider(str, Enum):
    GROQ = "groq"
    SAMBANOVA = "sambanova"


@dataclass
class ModelConfig:
    name: str
    temperature: float
    provider: ModelProvider


LLAMA_3_GROQ = ModelConfig("llama-3.3-70b-versatile", 0.0, ModelProvider.GROQ)
# DEEPSEEK_R1 = ModelConfig("DeepSeek-R1", 0.0, ModelProvider.SAMBANOVA)
DEEPSEEK_R1 = ModelConfig("DeepSeek-R1-Distill-Llama-70B", 0.0, ModelProvider.SAMBANOVA)


class Config:
    SEED = 42
    MODEL = LLAMA_3_GROQ
    COMPLEX_MODEL = DEEPSEEK_R1

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        DATA_DIR = APP_HOME / "data"
        DATABASE_PATH = DATA_DIR / "ecommerce.sqlite"


def seed_everything(seed: int = Config.SEED):
    random.seed(seed)
