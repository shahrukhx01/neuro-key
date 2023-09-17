import pathlib
from typing import List

import hydra
import srsly
from hydra.utils import instantiate
from omegaconf import DictConfig

from neuro_key.data.document import Document

DATASET_PATH = pathlib.Path(__file__).parent.parent.parent.resolve() / "sample_data" / "sample_dataset.jsonl"
CONFIG_PATH = pathlib.Path(__file__).parent.resolve() / "configs"
CONFIG_NAME = {"embed_rank": "embed_rank_config.yaml", "embed_rank++": "embed_rank++_config.yaml"}


class EmbedRank:
    def __init__(self, config_path: str, config_name: str):
        self._init_config(config_path, config_name)
        self._pipeline = instantiate(self._config)

    def _init_config(self, config_path: str, config_name: str) -> None:
        """Init the config file.

        Args:
            config_path: The path to the config file.
            config_name: The name of the config file.
        """

        @hydra.main(config_path=config_path, config_name=config_name)
        def load_config(config: DictConfig) -> None:
            self._config = config

        load_config()

    def extract_keywords(self, dataset: List[Document]) -> List[Document]:
        """Extract keywords from a dataset.

        Args:
            dataset: A list of documents.

        Returns:
            A list of documents with keywords.
        """
        result: List[Document] = dataset
        for node in self._pipeline["nodes"]:
            result = node.run(result)
        print(result[0].keyphrase_candidates)
        return result


if __name__ == "__main__":
    # load dataset and pipeline
    dataset: List[Document] = [Document(**document) for document in srsly.read_jsonl(DATASET_PATH)]
    embed_rank_pipeline = EmbedRank(config_path=str(CONFIG_PATH), config_name=CONFIG_NAME["embed_rank++"])

    # execute pipeline
    embed_rank_pipeline.extract_keywords(dataset=dataset)
