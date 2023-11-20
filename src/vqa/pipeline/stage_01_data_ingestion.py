from vqa.config.configuration import ConfigurationManager
from vqa.components.data_ingestion import DataIngestion
from vqa import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        data_ingestion_input_images_train_config = config.get_data_ingestion_input_images_train_config()
        data_ingestion_input_images_validation_config = config.get_data_ingestion_input_images_validation_config()
        data_ingestion_input_images_test_config = config.get_data_ingestion_input_images_test_config()
        data_ingestion_input_questions_train_config = config.get_data_ingestion_input_questions_train_config()
        data_ingestion_input_questions_validation_config = config.get_data_ingestion_input_questions_validation_config()
        data_ingestion_input_questions_test_config = config.get_data_ingestion_input_questions_test_config()
        data_ingestion_annotations_train_config = config.get_data_ingestion_annotations_train_config()
        data_ingestion_annotations_validation_config = config.get_data_ingestion_annotations_validation_config()

        all_datas = [data_ingestion_input_images_train_config, data_ingestion_input_images_validation_config, data_ingestion_input_images_test_config,
                     data_ingestion_input_questions_train_config, data_ingestion_input_questions_validation_config, data_ingestion_input_questions_test_config,
                     data_ingestion_annotations_train_config, data_ingestion_annotations_validation_config]

        for cfg in all_datas:
            self.download_and_unzip(cfg)

        # self.download_and_unzip(data_ingestion_annotations_train_config)
            

    def download_and_unzip(self, config_path):
        data_ingestion = DataIngestion(config=config_path)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e