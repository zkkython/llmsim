from huggingface_hub import hf_hub_download


class HuggingFaceConfigsLoader:

    def download_configs_from_hugging_face(self, path: str) -> str:
        """
        @params path:
        """
        local_dir_path = "hf_config/huggingface.co/" + path
        return hf_hub_download(
            repo_id=path, filename="config.json", local_dir=local_dir_path
        )
