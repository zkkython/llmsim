import logging

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    logging.warning(
        "huggingface_hub is not installed. Please install it to use HuggingFaceConfigsLoader."
    )


class HuggingFaceConfigsLoader:

    @staticmethod
    def download_configs_from_hugging_face(path: str) -> str:
        """
        @params path:
        """
        local_dir_path = "hf_config/huggingface.co/" + path
        return hf_hub_download(
            repo_id=path, filename="config.json", local_dir=local_dir_path
        )
