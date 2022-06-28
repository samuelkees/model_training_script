import click
import torch
import json
import os

from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Average, Accuracy
from ignite.handlers import Events, EpochOutputStore, EarlyStopping

from src.training._script_util import (register_ss_index, retrieve_ss_index, gen_new_ssindex, 
                          _update_ssindex, _transform_labels, load_ss_index, gen_logo_dataloader, 
                          gen_logo_triplet_dataloader, _register_model, toggel_freeze, 
                          get_param_to_train, _get_special_classes, flatten, TrainConf, score_factory)
from src.training._model import  TripletLogoModel, DoubleTripledLoss
from src.training._ignite import (inference_factory, logo_prepair_batch, train_factory, 
                     logo_triplet_prepair_batch, TripletAccuracy)
from src.training._azure_util import azure_log_metrics


@click.command()
@click.option("--ss_index_name", default="", type=str, help="")
@click.option("--ss_index_version", default=-1, type=int, help="")
@click.option("--model_name", default="", type=str, help="")
@click.option("--model_version", default=-1, type=int, help="")
@click.option("--validation_dataset_path", default="./notebooks/azure/data/val_test_df_klein.parquet", type=str, help="")
@click.option("--test_dataset_path", default="", type=str, help="")
@click.option("--train_dataset_path", default= "./notebooks/azure/data/script_test_data.parquet" ,type=str, help="")
@click.option("--root_data_path", default= "/home/schulz-kees/data/datasets/logo_dataset/data", type=str, help="")
@click.option("--device", type=str, default="cuda", help="cuda or cpu")
@click.option("--batch_size", default=6, type=int, help="cuda or cpu") # 35 is good for azure
@click.option("--num_workers", default=4, type=int, help="cuda or cpu")
@click.option("--k", type=int, default=30, help="number of neighbors to search (set it hiher then the number for items per class)")
@click.option("--gamma", type=float, default=0.5,  help="parameter to controll the leverage of the distance")
@click.option("--lr", type=float, default=0.001,  help="the learning rate")
@click.option("--margin", type=float, default=0,  help="the margin for the triplet loss")
@click.option("--max_loss", type=float, default=0.01,  help="the max number the triplet loss can reach")
@click.option("--num_samples_epoch", type=int, default=20,  help="number of samples in each epoch")
@click.option("--max_epochs", type=int, default=100,  help="number of epochs")
@click.option("--interval_update_index", type=int, default=2,  help="number of epochs")
@click.option("--max_dis", type=float, default=500,  help="the max dis of image in the ss index to consider for classification")
@click.option("--loss_weight", type=float, default=0.75,  help="wight for image and geomertry vector [0-1]")
@click.option("--freeze", type=int, default=11,  help="number of model layer not to freeze")
def main(**kwargs):
    # generate config
    CONF = TrainConf(**kwargs)
    # load model
    model = CONF.model
    toggel_freeze(model, CONF._conf.freeze, False)
    print("model parameter to train ", get_param_to_train(model))
    
    # dataloader train
    train_meta, _, train_loader = gen_logo_dataloader(CONF._conf.train_dataset_path, CONF._conf.root_data_path, 
                                                      CONF._conf.batch_size, CONF._conf.num_workers)

    # dataloader val 
    val_meta, _, val_loader = gen_logo_dataloader(CONF._conf.validation_dataset_path, CONF._conf.root_data_path, 
                                                      CONF._conf.batch_size, CONF._conf.num_workers)
     # load index
    print("get ssindex...")

    index_engine = Engine(inference_factory(model,CONF._conf.device, None, None, None, logo_prepair_batch))
    ProgressBar().attach(index_engine)
    eos = EpochOutputStore()
    eos.attach(index_engine)
    if CONF.ssindex is None:
        ss_index, tensors = gen_new_ssindex(train_loader, train_meta, index_engine, eos, "class_id")
        register_ss_index(train_meta, tensors, f"./outputs/{CONF.run._run_id}", CONF.ws)
    else:
        ss_index = CONF.ssindex
    print("get ssindex done..")
    # triplet_data loader
    train_triplet_loader = gen_logo_triplet_dataloader(ss_index, CONF._conf.k, CONF._conf.num_samples_epoch, 
                                                       CONF._conf.root_data_path, CONF.triplet_batch_size, CONF._conf.num_workers)
    print("start training...")
    ### train ###
    # define triplet model for training
    t_model = TripletLogoModel(model)
    model.to(CONF._conf.device)
    # set optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            model.parameters()), lr=CONF._conf.lr)
    # set loss fun
    loss_fn = DoubleTripledLoss(max_value= CONF._conf.max_loss, margin=CONF._conf.margin, weight=CONF._conf.loss_weight)
    # train step:
    train_func = train_factory(t_model, optimizer, loss_fn, CONF._conf.device, logo_triplet_prepair_batch)
    train_engine = Engine(train_func)
    # progress bar for training
    ProgressBar().attach(train_engine)
    # define metrics
    TripletAccuracy(CONF._conf.margin, lambda x: x['embeddings']["x"]).attach(train_engine, 'image_triplet_acc')
    TripletAccuracy(CONF._conf.margin, lambda x: x['embeddings']["y"]).attach(train_engine, 'geomertry_triplet_acc')
    Average(lambda x: x['loss']).attach(train_engine, 'avg_loss')
    Average(lambda x: x['image_loss']).attach(train_engine, 'avg_image_loss')
    Average(lambda x: x['geomertry_loss']).attach(train_engine, 'avg_geomertry_loss')
    # log train metrics
    train_engine.add_event_handler(Events.EPOCH_COMPLETED, azure_log_metrics, CONF.run)

    # update ssindex and register ssindex and model 
    train_engine.add_event_handler(Events.EPOCH_COMPLETED(every=CONF._conf.interval_update_index), 
                                   _register_model, model,  CONF.run, CONF.ws)
    train_engine.add_event_handler(Events.EPOCH_COMPLETED(every=CONF._conf.interval_update_index), 
                                   _update_ssindex, index_engine, eos, train_loader, 
                                   train_meta, CONF.run, CONF.ws, update_index= True)

    # save "big triplet" ; triplet with a big loss number
   # @train_engine.on(Events.ITERATION_COMPLETED)
    def save_big_triplets(engine):
        if engine.state.output["big_triplet"]:
            try:
                os.makedirs("./outputs")
            except OSError:
                pass
            with open("./outputs/big_triplet.txt", "a") as f:
                for data in engine.state.output["big_triplet"]:
                    data['epoch'] = engine.state.epoch
                    f.write(json.dumps(data) + "\n")

    # EarlyStopping_event for class_acc == 0
    early_stopping = EarlyStopping(patience=2, score_function=score_factory(train_engine), trainer=train_engine)

    # validate train
    @train_engine.on(Events.EPOCH_COMPLETED(every=CONF._conf.interval_update_index))
    def _evaluate_event(engine):
        x = engine.state.epoch 
        dataloader = {"validate": val_loader}
        try:
            print("retrieve_model:")
            ssindex, azure_model = retrieve_ss_index(CONF.run._run_id, None, CONF.ws, "class_id")
        except Exception as e:
            print(e)
            print("azure offline run mode?")
            ssindex = load_ss_index(f"./outputs/{CONF.run._run_id}", "class_id")

        # def index_evaluator
        index_evaluator = Engine(inference_factory(model,CONF._conf.device, 
                                    ssindex, CONF._conf.k, CONF._conf.gamma, 
                                    logo_prepair_batch, max_dis=CONF._conf.max_dis))
        Accuracy(_transform_labels).attach(index_evaluator, "class_acc")
        special_classes_store = EpochOutputStore(_get_special_classes)
        special_classes_store.attach(index_evaluator)
        ProgressBar().attach(index_evaluator)

        # add earlystipping event to index_evaluator
        index_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

        @index_evaluator.on(Events.COMPLETED)
        def _update_special_classes(_):
            train_engine.state.dataloader.dataset._special_classes = flatten(special_classes_store.data)
            print("set_special_classes:", set(flatten(special_classes_store.data)))
            train_engine.set_data(train_engine.state.dataloader)


        @index_evaluator.on(Events.ITERATION_COMPLETED)
        def save_big_dis(engine):
            for x, y in zip(engine.state.output["y_pred"], engine.state.output["y"]):
                if len(x) == 3:
                    try:
                        os.makedirs("./outputs")
                    except OSError:
                        pass
                    with open("./outputs/y_pred.txt", "a") as f:
                        data = {}
                        # float not json serializable (hotfix)
                        x = (x[0], str(x[1]), [(a, str(b)) for a,b in x[2]])
                        data["y_pred"] = x
                        data['epoch'] = train_engine.state.epoch
                        data['y'] = y
                        f.write(json.dumps(data) + "\n")

        # run index_evaluator
        for name, loader in dataloader.items():
            with index_evaluator.add_event_handler(Events.COMPLETED, azure_log_metrics, 
                                                    CONF.run, f"{name}_", x):
                print("validate index:")
                index_evaluator.run(loader)
        try:
            print("update model tags:")
            azure_model.update(tags=index_evaluator.state.metrics)
        except Exception as e:
            print(e)
            print("azure offline run mode?")

    # run training
    train_engine.run(train_triplet_loader, CONF._conf.max_epochs)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("script failed..")
        raise e
