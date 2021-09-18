from preprocessor import PreprocessingData
from config import PipelineConfig
from mgf_parser import MGFParser
from db_manager import MongoDBManager, MixedData
from datasets import ModelData
from resnet_model import multi_scale
from process_models import ModelProcessor
from metrics import Evaluator
from utils import ModelOutData
from feature_extractor import FeaturesExtractor


class PipelineProcessor(object):
    def __init__(self, config_path):
        self.config = PipelineConfig(config_path)
        self.preprocessor = PreprocessingData()
        self.data = None

    def process_pipeline(self):
        assert self.config.pipeline_steps
        for step in self.config.pipeline_steps:
            self.pipeline_step(step)

    def pipeline_step(self, step):
        if step == 'load from folder':
            print(f'Load data from {self.config.data_folder}')
            parser = MGFParser()
            parser.load_directory(self.config.data_folder,
                                  labels=self.config.labels,
                                  val_len=self.config.val_len)
            self.data = parser.data

        if step == 'clean-create':
            if self.config.use_elliptic:
                print(f'Cleaning data (elliptic envelope...)')
                self.preprocessor.create_outlier(self.data, contamination=self.config.contamination)
                self.preprocessor.elliptic_envelope(self.data)
                if self.config.save_preprocessing_models:
                    self.preprocessor.save_outlier_detector(self.config.outlier_model_path)

        if step == 'scale-create':
            if self.config.mm_scale:
                print('Scale data --MinMaxScaler...')
                self.preprocessor.create_scale_object(self.data, log=self.config.log)
                self.preprocessor.mm_scale(self.data, log=self.config.log)
                if self.config.save_preprocessing_models:
                    self.preprocessor.save_mm_scale(self.config.mm_models_path)

        if step == 'align':
            print('Align data...')
            self.preprocessor.align_data_to_rti(self.data,
                                                self.config.rti_align_bound,
                                                self.config.rti_align_step)
        if step == 'upload to db':
            if self.config.db_type == 'mongo':
                db_manager = MongoDBManager(config=self.config)
                print(f'Upload data {self.config.mongo_db_name}-->{self.config.mongo_col_name}...')
                db_manager.connect()
                db_manager.upload_data(self.data)

        if step == 'load from db':
            if self.config.db_type == 'mongo':
                db_manager = MixedData(config=self.config)
                print(f'Load data from {self.config.mongo_db_name}-->{self.config.mongo_col_name}...')
                db_manager.connect()
                db_manager.get_data()
                self.data = db_manager.dataset
                assert self.data
                model_data = ModelData(self.data)
                model_data.create_model_data()

        if step == "load from file":
            mgf_parser = MGFParser()
            file_ = self.config.file_path
            assert file_
            mgf_parser.load_mgf(file_, label=-1, is_test=1)
            self.data = mgf_parser.data
            if self.config.save_output and self.config.output_dir:
                for key in self.data:
                    item = self.data[key].get('data')
                    file_ = item.get('file')
                    out_data = ModelOutData(filename=file_, out_dir=self.config.output_dir)
                    out_data.save_input_data(data=item)

        if step == 'clean-load':
            assert self.data
            print(f'Cleaning data (elliptic envelope...)')
            self.preprocessor.load_outlier_detector(self.config.outlier_model_path)
            self.preprocessor.elliptic_envelope(self.data)
            if self.config.save_output and self.config.output_dir:
                for key in self.data:
                    item = self.data[key].get('data')
                    file_ = item.get('file')
                    out_data = ModelOutData(filename=file_, out_dir=self.config.output_dir)
                    out_data.save_input_data(data=item, mode='clean')

        if step == 'scale-load':
            print('Scale data --MinMaxScaler...')
            assert self.data
            self.preprocessor.load_mm_scale(self.config.mm_models_path)
            self.preprocessor.mm_scale(self.data)

        if step == 'train':
            print("Start training mode...")
            model_data = ModelData(self.data)
            model_data.create_model_data()
            train_dataset = model_data.train_loader
            test_dataset = model_data.test_loader
            assert train_dataset
            assert test_dataset
            if self.config.model_type == 'Resnet':
                model = multi_scale()#MSResNet(input_channel=2, layers=[1, 1, 1, 1], num_classes=6)
                # model = ResNet1D(
                #     in_channels=2,
                #     base_filters=64,
                #     kernel_size=4,
                #     stride=2,
                #     groups=1,
                #     n_classes=len(self.config.labels),
                #     n_block=23
                #)

            assert model
            evaluator = None

            if self.config.logging:
                evaluator = Evaluator(console=self.config.console,
                                      confusion=self.config.confusion_matrix,
                                      classification=self.config.classification_report,
                                      labels=self.config.labels
                                      )

            model_processor = ModelProcessor(model=model,
                                             train_loader=train_dataset,
                                             test_loader=test_dataset,
                                             evaluator=evaluator)
            model_processor.train_model(epoch=self.config.epochs,
                                        save_best_model=self.config.best_model,
                                        model_path=self.config.model_path
                                        )

        if step == 'validate':
            assert self.data
            print("Start validation mode...")
            feature_extractor = None
            evaluator = None
            model_data = ModelData(self.data)
            model_data.create_model_data()
            test_dataset = model_data.val_loader
            if self.config.logging:
                evaluator = Evaluator(console=self.config.console,
                                      confusion=self.config.confusion_matrix,
                                      classification=self.config.classification_report,
                                      labels=self.config.labels,
                                      save_output=self.config.save_output,
                                      output_dir=self.config.output_dir
                                      )

            if self.config.model_type == 'Resnet':
                model = multi_scale()
            assert model
            if self.config.save_output and self.config.output_dir and self.config.use_cam:
                assert self.config.cam_layer
                item = self.data[0].get('data')
                pm = item.get('pm')
                data_size = len(pm)
                feature_extractor = FeaturesExtractor(model, self.config.cam_layer, data_size=data_size)

            model_processor = ModelProcessor(model=model,
                                             train_loader=None,
                                             test_loader=test_dataset,
                                             evaluator=evaluator,
                                             feature_extractor=feature_extractor)
            model_processor.load_model(self.config.model_path)
            model_processor.test_model()
