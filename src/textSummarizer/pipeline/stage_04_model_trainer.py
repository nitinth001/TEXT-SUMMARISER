from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging import logger
class ModelTrainerTrainingPipeline:
    def _init_(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()

            # Get config
            model_trainer_config = config.get_model_trainer_config()

            # Initialize trainer
            model_trainer = ModelTrainer(config=model_trainer_config)

            # Start training
            model_trainer.train()

        except Exception as e:
            logger.exception(e)
            raise e