import argparse
import torch
import os
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders,create_samplers
from recbole.data.dataset import KnowledgeBasedDataset
from recbole.sampler import KGSampler
from recbole.data.dataloader import KnowledgeBasedDataLoader,NegSampleEvalDataLoader
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
from trainer import MultiKGTrainer
import importlib

def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True,pretrained_file=None):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    # dataset filtering
    # dataset = create_dataset(config)
    os.popen(f'cp {config["data_path"]}/{config["dataset"]}.kg.meta {config["data_path"]}/{config["dataset"]}.kg')
    dataset = KnowledgeBasedDataset(config)
    logger.info(dataset)
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)
    kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
    train_data_1 = KnowledgeBasedDataLoader(config, train_dataset, train_sampler, kg_sampler, shuffle=True)
    valid_data = NegSampleEvalDataLoader(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = NegSampleEvalDataLoader(config, test_dataset, test_sampler, shuffle=False)
    os.popen(f'cp {config["data_path"]}/{config["dataset"]}.kg.{config["cs_kg_name"]} {config["data_path"]}/{config["dataset"]}.kg')
    dataset = KnowledgeBasedDataset(config)
    logger.info(dataset)
    built_datasets = dataset.build()
    train_dataset, valid_dataset, test_dataset = built_datasets
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)
    kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
    train_data_2 = KnowledgeBasedDataLoader(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(model)(config, train_data_1.dataset,train_data_2.dataset).to(config['device'])
    logger.info(model)
    
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # trainer loading and initialization
    trainer = MultiKGTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data_1,train_data_2, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """

    model_file_name = model_name.lower()
    model_module = None
    module_path = '.'.join(['model', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CKE', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Electronics', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='props/run_csrec.yaml', help='config files')
    parser.add_argument('--pretrain_file', '-p', type=str, help='pretrain file', default='')

    args, _ = parser.parse_known_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    baseline_func = run_recbole

    baseline_func(model=args.model, dataset=args.dataset, config_file_list=config_file_list,pretrained_file=args.pretrain_file)
