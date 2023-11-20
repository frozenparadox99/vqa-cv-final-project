from vqa.constants import *
import os
from pathlib import Path
from vqa.utils.common import read_yaml, create_directories
from vqa.entity.config_entity import (DataIngestionConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_input_images_validation_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_input_images_validation

        create_directories([config.root_dir])

        data_ingestion_input_images_validation_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_input_images_validation_config
    
    def get_data_ingestion_input_images_train_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_input_images_train

        create_directories([config.root_dir])

        data_ingestion_input_images_train_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_input_images_train_config
    
    def get_data_ingestion_input_images_test_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_input_images_test

        create_directories([config.root_dir])

        data_ingestion_input_images_test_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_input_images_test_config
    
    def get_data_ingestion_input_questions_train_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_input_questions_train

        create_directories([config.root_dir])

        data_ingestion_input_questions_train_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_input_questions_train_config
    
    def get_data_ingestion_input_questions_validation_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_input_questions_validation

        create_directories([config.root_dir])

        data_ingestion_input_questions_validation_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_input_questions_validation_config
    
    def get_data_ingestion_input_questions_test_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_input_questions_test

        create_directories([config.root_dir])

        data_ingestion_input_questions_test_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_input_questions_test_config
    
    def get_data_ingestion_annotations_train_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_annotations_train

        create_directories([config.root_dir])

        data_ingestion_annotations_train_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_annotations_train_config
    
    def get_data_ingestion_annotations_validation_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_annotations_validation

        create_directories([config.root_dir])

        data_ingestion_annotations_validation_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_annotations_validation_config