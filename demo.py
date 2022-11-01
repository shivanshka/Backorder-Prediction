from Backorder.pipeline.training_pipeline import Training_Pipeline
from Backorder.logger import logging

def main():
    try:
        train = Training_Pipeline()
        train.run_training_pipeline()
    except Exception as e:
        logging.error(e)

if __name__=="__main__":
    main()