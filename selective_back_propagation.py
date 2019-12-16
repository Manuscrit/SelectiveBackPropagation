import collections

import numpy as np
import torch
# from loguru import logger


class SelectiveBackPropagation:
    """
    Selective_Backpropagation from paper Accelerating Deep Learning by Focusing on the Biggest Losers
    https://arxiv.org/abs/1910.00762v1
    Without:
            y_pred = model(x)
            loss = criterion(y_pred, y).mean()
            loss.backward()  
    With: 
            with torch.no_grad():
                y_pred = model(x)
            not_reduced_loss = criterion(y_pred, y)
            loss = selective_bp.selective_back_propagation(not_reduced_loss, x, y)
    """
    def __init__(self, compute_losses_func, update_weights_func, optimizer, model,
                 batch_size, epoch_length, loss_selection_threshold=False):

        self.loss_selection_threshold = loss_selection_threshold
        self.compute_losses_func = compute_losses_func
        self.update_weights_func = update_weights_func
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.model = model

        self.loss_hist = collections.deque([], maxlen=batch_size*epoch_length)
        self.selected_inputs, self.selected_targets = [], []

    def selective_back_propagation(self, loss_per_img, data, targets):
        effective_batch_loss = None

        cpu_losses = loss_per_img.detach().clone().cpu()
        self.loss_hist.extend(cpu_losses.tolist())
        np_cpu_losses = cpu_losses.numpy()
        selection_probabilities = self._get_selection_probabilities(np_cpu_losses)

        selection = selection_probabilities > np.random.random(*selection_probabilities.shape)

        if self.loss_selection_threshold:
            higher_thres = np_cpu_losses > self.loss_selection_threshold
            selection = np.logical_or(higher_thres, selection)

        selected_losses = []
        for idx in np.argwhere(selection).flatten():
            selected_losses.append(np_cpu_losses[idx])

            self.selected_inputs.append(data[idx, ...].detach().clone())
            self.selected_targets.append(targets[idx, ...].detach().clone())
            if len(self.selected_targets) == self.batch_size:
                self.model.train()
                predictions = self.model(torch.stack(self.selected_inputs))
                effective_batch_loss = self.compute_losses_func(predictions,
                                                                torch.stack(self.selected_targets))
                self.update_weights_func(effective_batch_loss)
                effective_batch_loss = effective_batch_loss.mean()
                self.model.eval()
                self.selected_inputs = []
                self.selected_targets = []

        # logger.info("Mean of input loss {}".format(np.array(np_cpu_losses).mean()))
        # logger.info("Mean of loss history {}".format(np.array(self.loss_hist).mean()))
        # logger.info("Mean of selected loss {}".format(np.array(selected_losses).mean()))
        # logger.info("Mean of effective_batch_loss {}".format(effective_batch_loss))
        return effective_batch_loss

    def _get_selection_probabilities(self, loss):
        percentiles = self._percentiles(self.loss_hist, loss)
        return percentiles ** 2

    def _percentiles(self, hist_values, values_to_search):
        # TODO Speed up this again. There is still a visible overhead in training. 
        hist_values, values_to_search = np.asarray(hist_values), np.asarray(values_to_search)

        percentiles_values = np.percentile(hist_values, range(100))
        sorted_loss_idx = sorted(range(len(values_to_search)), key=lambda k: values_to_search[k])
        counter = 0
        percentiles_by_loss = [0] * len(values_to_search)
        for idx, percentiles_value in enumerate(percentiles_values):
            while values_to_search[sorted_loss_idx[counter]] < percentiles_value:
                percentiles_by_loss[sorted_loss_idx[counter]] = idx
                counter += 1
                if counter == len(values_to_search) : break
            if counter == len(values_to_search) : break
        return np.array(percentiles_by_loss)/100
