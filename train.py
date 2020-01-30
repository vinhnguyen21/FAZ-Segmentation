import argparse
import json

from src.trainer import FAZSegmentation

def main(args):
    # Parse config
    params = json.load(open(args.config, 'r'))
    print(params)
    trainer = FAZSegmentation(**params)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2DS - OCTA segmentation - Training')
    parser.add_argument('--config', default='./config/train_config.json', type=str, help='config file')
    args = parser.parse_args()
    main(args)
