import torch.cuda
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import net
import os
import utils
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

data_path = './dataset/'
model_dir = './experiments/base_model'


def train(model, optimizer, loss_fn, data_iterator, writer,metrics, params, epoch):
    model.train()

    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(data_iterator)) as t:
        for i, (train_batch, labels_batch) in enumerate(data_iterator):
            if params.device == 'cuda:0':
                # train_batch, labels_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
                train_batch, labels_batch = train_batch.to(params.device, non_blocking=True), labels_batch.to(params.device, non_blocking=True)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

                writer.add_scalar('Loss/train', summary_batch['loss'], epoch * len(data_iterator) + i)
                writer.add_scalar('Accuracy/train', summary_batch['accuracy'], epoch * len(data_iterator) + i)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    print("- Train metrics: " + metrics_string)


def evaluate(model, loss_fn, data_iterator, metrics, params):
    model.eval()
    summ = []

    for data_batch, labels_batch in data_iterator:
        if params.device == 'cuda:0':
            data_batch, labels_batch = data_batch.to(params.device, 
                non_blocking=True), labels_batch.to(params.device, non_blocking=True)
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    print("- Eval metrics : " + metrics_string)
    return metrics_mean

def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, writer, metrics, params, restore_file=None):
    if restore_file is not None:
        restore_path = os.path.join(
            model_dir, restore_file + '.pth.tar')
        print("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
    
    best_val_acc = 0.0

    for epoch in range(params.num_epochs):  # TODO: while True
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        num_steps = (params.train_size + 1) // params.batch_size
        train(model, optimizer, loss_fn, train_loader, writer, metrics, params, epoch)
        val_metrics = evaluate(model, loss_fn, val_loader, metrics, params)
        val_acc = val_metrics['accuracy']
        writer.add_scalar('Loss/evaluate', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/evaluate', val_acc, epoch)
        is_best = val_acc >= best_val_acc
        # whatever acc is best ornot, save checkpoint
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)
        if is_best:
            # best result save json
            print('- Found new best acc')
            best_val_acc = val_acc
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
        # currently result save json
        last_json_path = os.path.join(model_dir, 'metrics_val_last_weights.json')
        utils.save_dict_to_json(val_metrics, last_json_path)

def run():
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(230)
    if params.device == 'cuda:0':
        torch.cuda.manual_seed(230)
    utils.set_logger(os.path.join(model_dir, 'train.log'))
    writer = SummaryWriter(log_dir=os.path.join(model_dir, 'tensorboard_logs'))
    print("Loading the datasets...")

    train_dataset = torchvision.datasets.MNIST(root=data_path, train=True,
                    transform=torchvision.transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_path, train=False,
                   transform=torchvision.transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size)
    val_loader = DataLoader(test_dataset, batch_size=params.batch_size)
    params.train_size = len(train_dataset)
    params.val_size = len(test_dataset)

    print("- done.")
    model = net.Model().cuda() if params.device == 'cuda:0' else net.Model()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    print("Starting training for {} epoch(s)".format(params.num_epochs))
    metrics = net.metrics
    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, writer, metrics, params)

    writer.close()

if __name__ == '__main__':
    # TODO:get parse_args()
    run()