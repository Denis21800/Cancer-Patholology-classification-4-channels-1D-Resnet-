from utils import singleton
import json


@singleton
class PipelineConfig(object):
    def __init__(self, config_path):
        assert config_path
        self.config_path = config_path
        self.json_object = json.load(open(self.config_path, 'r'))
        labels = self.json_object.get("labels")
        self.labels = {int(key): labels[key] for key in labels}
        assert self.labels
        self.pipeline_steps = self.json_object.get("pipeline steps")
        assert self.pipeline_steps
        self.db_type = self.json_object.get('db_type')
        self.__load_data_params()
        if self.db_type == 'mongo':
            self.__load_mongo_params()

        self.__load_train_params()
        self.__load_test_params()
        self.__load_preprocessing_params()
        self.__load_logging_params()

    def __load_data_params(self):
        env_config = self.json_object["load from folder"]
        self.data_folder = env_config.get("data folder")
        self.val_len = env_config.get("validation")
        self.file_path = env_config.get("file_path")

    def __load_preprocessing_params(self):
        env_config = self.json_object["preprocessing"]
        self.use_elliptic = env_config.get("elliptic")
        self.contamination = env_config.get("contamination")
        self.mm_scale = env_config.get("min-max_scale")
        self.rti_align = env_config.get("rti_align")
        self.log = env_config.get("log")
        self.rti_align_bound = env_config.get("align_bound")
        self.rti_align_step = env_config.get("align_step")
        self.save_preprocessing_models = env_config.get("save_models")
        self.mm_models_path = env_config.get("mm_model_path")
        self.outlier_model_path = env_config.get("outlier_model_path")

    def __load_mongo_params(self):
        env_config = self.json_object["mongo_params"]
        self.mongo_db_name = env_config.get("db_name")
        self.mongo_host = env_config.get("host")
        self.mongo_port = env_config.get("port")
        self.mongo_col_name = env_config.get("col_name")

    def __load_train_params(self):
        env_config = self.json_object["train model"]
        self.model_path = env_config.get("model_path")
        self.model_type = env_config.get("model_type")
        self.epochs = env_config.get("n_epochs", 50)
        self.best_model = env_config.get("save_best_model")

    def __load_test_params(self):
        env_config = self.json_object["test model"]
        self.model_path = env_config.get("model_path")
        self.model_type = env_config.get("model_type")
        self.save_output = env_config.get("save_output")
        self.output_dir = env_config.get("output_dir")
        self.use_cam = env_config.get("cam_extractor")
        self.cam_layer = env_config.get("cam_layer")

    def __load_logging_params(self):
        env_config = self.json_object["logging"]
        self.logging = env_config.get("logging_results")
        self.console = env_config.get("console")
        self.use_tensorboard = env_config.get("use_tensorboard")
        self.classification_report = env_config.get("classification_report")
        self.confusion_matrix = env_config.get("confusion_matrix")
