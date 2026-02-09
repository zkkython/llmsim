import logging


class RemoteConfigsLoader:
    """
    Supports loading the model's config.json file from a remote location.
    Usage:
        Specify to use huggingface/modelscope, or automatically load the model configuration file from huggingface in the default mode.
        ```
        --model_path huggingface.co/zai-org/GLM-4.7-Flash
        --model_path zai-org/GLM-4.7-Flash # Automatically loads models from huggingface
        --model_path modelscope.cn/moonshotai/Kimi-K2.5
        ```

    Directly read the local configuration file:
        Directly load the local model configuration file.

        ```
        --model_path hf_config/deepseek_671b_r1_config.json
        ```
    """

    @staticmethod
    def load_configs_from_remote(path: str)->str:
        if path.startswith('huggingface.co') :
            # like "huggingface.co/zai-org/GLM-4.7-Flash"
            path = path.replace('huggingface.co/', '')
            return RemoteConfigsLoader.download_configs_from_hugging_face(path)
        elif path.startswith('modelscope.cn'):
            # like "modelscope.cn/moonshotai/Kimi-K2.5"
            path = path.replace('modelscope.cn/', '')
            return RemoteConfigsLoader.download_configs_from_model_scope(path)
        else:
            # like "zai-org/GLM-4.7-Flash"
            return RemoteConfigsLoader.download_configs_from_hugging_face(path)

    @staticmethod
    def download_configs_from_model_scope(path: str) -> str:
        try:
            local_dir_path = "hf_config/modelscope.cn/" + path
            from modelscope.hub.snapshot_download import snapshot_download
            aim_path = snapshot_download(
                model_id=path, allow_patterns="config.json", local_dir=local_dir_path
            )
            if aim_path is None or aim_path == '':
                raise ValueError('download model configs error for path : ' + path)
            return aim_path + "/config.json"
        except ImportError as e:
            logging.warning("huggingface_hub is not installed. Please install it to use RemoteLoader.")
            logging.warning("   for example: `pip install modelscope`.")
            raise e

    @staticmethod
    def download_configs_from_hugging_face(path: str) -> str:
        """
        @params path:
        """
        local_dir_path = "hf_config/huggingface.co/" + path
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id=path, filename="config.json", local_dir=local_dir_path
            )
        except ImportError as e:
            logging.warning("huggingface_hub is not installed. Please install it to use RemoteLoader.")
            logging.warning("   for example: `pip install huggingface-hub`.")
            raise e
