from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utils import ModelOutData
from graph_net import NetGraph


class Evaluator(object):
    def __init__(self,
                 labels=None,
                 console=True,
                 classification=True,
                 confusion=True,
                 use_tensorboard=True,
                 save_output=False,
                 output_dir=None,
                 feature_extractor=None):
        self.labels = labels
        self.use_tensorboard = use_tensorboard
        self.confusion = confusion
        self.classification = classification
        self.console = console
        self.classification = True
        self.confusion = True
        self.train_loses = 0
        self.eval_loses = 0
        self.train_examples = 0
        self.eval_examples = 0
        self.train_correct = 0
        self.eval_correct = 0
        self.eval_output_list = []
        self.eval_true_list = []
        self.best_model = False
        self.best_val_acc = 0
        self.epoch = 0
        self.epoch_train_acc = 0
        self.epoch_val_acc = 0
        self.tensorboard_writer = SummaryWriter() #if use_tensorboard else None
        self.save_output = save_output
        self.output_dir = output_dir
        self.feature_extractor = feature_extractor
        self.log_device = torch.device('cpu')
        self.log_probability = {}

    def log_train_iter(self, model_output, y_target, train_loss):
        with torch.no_grad():
            correct = torch.eq(torch.max(softmax(model_output, dim=1), dim=1)[1], y_target).view(-1)
            correct = correct.to(self.log_device).detach()
            loss = train_loss.to(self.log_device).detach()
            self.train_correct += torch.sum(correct).item()
            self.train_examples += correct.shape[0]
            self.train_loses += loss.cpu().numpy()
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Train loses', loss, self.epoch)

    def log_eval_iter(self, model_output, y_target, eval_loss):
        with torch.no_grad():
            correct = torch.eq(torch.max(softmax(model_output, dim=1), dim=1)[1], y_target).view(-1)
            predicted_label = model_output.data.cpu().numpy().argmax()
            correct = correct.to(self.log_device).detach()
            loss = eval_loss.to(self.log_device).detach()
            self.eval_correct += torch.sum(correct).item()
            actual_label = y_target.cpu().numpy()[0]
            self.eval_true_list.append(actual_label)
            self.eval_output_list.append(predicted_label)
            self.eval_examples += correct.shape[0]
            self.eval_loses += eval_loss.cpu().numpy()
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Eval loses', loss, self.epoch)

    def calculate_epoch_metrics(self):
        if self.train_examples == 0 or self.eval_examples == 0:
            return

        self.epoch_train_acc = round(self.train_correct / self.train_examples, 4)
        self.epoch_val_acc = round(self.eval_correct / self.eval_examples, 4)
        self.eval_loses = round(self.eval_loses / self.eval_examples, 4)
        self.train_loses = round(self.train_loses / self.train_examples, 4)
        if self.epoch_val_acc >= self.best_val_acc:
            self.best_val_acc = self.epoch_val_acc
            self.best_model = True
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Eval Accuracy', self.epoch_val_acc, self.epoch)
            self.tensorboard_writer.add_scalar('Train Accuracy', self.epoch_train_acc, self.epoch)

    def print_epoch_metrics(self):
        self.calculate_epoch_metrics()
        if self.console:
            headers = [f'Epoch: {self.epoch}', 'TRAIN', 'TEST']
            result_table = [
                ['Correct:', f'{self.train_correct}/{self.train_examples}',
                 f'{self.eval_correct}/{self.eval_examples}'],
                ['Accuracy:', self.epoch_train_acc, self.epoch_val_acc],
                ['Loses:', self.train_loses, self.eval_loses]
            ]

            print(tabulate(result_table, headers=headers))
            self.print_eval_reports()
            print('*' * 0xA1)

    def is_best_model(self):
        return self.best_model

    def reset(self):
        self.train_loses = 0
        self.eval_loses = 0
        self.train_examples = 0
        self.eval_examples = 0
        self.train_correct = 0
        self.eval_correct = 0
        self.best_model = False
        self.eval_true_list.clear()
        self.eval_output_list.clear()

    def next_epoch(self):
        self.reset()
        self.epoch += 1

    def __log_probability_update(self, probability, actual_class):
        data = self.log_probability.get(actual_class)
        p_ = probability.cpu().numpy()
        if not data:
            data = {'p_': p_, 'count': 1}
        else:
            count = data.get('count')
            p_log = data.get('p_')
            p_log += p_
            count += 1
            data.update({'p_': p_log, 'count': count})
        self.log_probability.update({actual_class: data})

    def get_test_params(self, model_output, y_target):
        probability_arr = softmax(model_output, dim=1).data.squeeze().cpu()
        class_idx = torch.topk(probability_arr, 1)[1].int().numpy()[0]
        probability_ = float(probability_arr.max().numpy())
        actual_class = y_target.cpu().numpy()[0]
        self.__log_probability_update(probability_arr, actual_class)
        result = class_idx == actual_class
        self.eval_true_list.append(actual_class)
        self.eval_output_list.append(class_idx)
        return probability_, class_idx, actual_class, result

    def log_test_iter(self,
                      model_output,
                      y_target,
                      file=None,
                      print_result=True,
                      feature_extractor=None,
                      input_data_index=None):
        p_, class_idx, actual_class, result = self.get_test_params(model_output, y_target)
        file_ = file[0]
        data = {'File': f"{file_ if file else 'N/D'}",
                'Actual class:': f"{self.labels.get(actual_class) if actual_class != - 1 else 'N/D'}",
                'Result': f"{result if actual_class != -1 else 'N/D'}",
                'Predicted class': f'{self.labels.get(class_idx)}',
                'Probability': f'{round(p_, 4)}'}
        if print_result:
            headers = []
            result_table = [[f'{key}:', data.get(key)] for key in data]
            print(tabulate(result_table, headers=headers))

        if self.save_output and self.output_dir:
            model_out_data = ModelOutData(self.output_dir, file_)
            model_out_data.save_prediction_result(data)
            if feature_extractor and input_data_index is not None:
                cam = feature_extractor.get_cam(model_output=model_output)
                model_out_data.save_cam(cam, input_data_index)

    def print_eval_reports(self):
        if self.classification:
            try:
                print('Classification report (Evaluation data):')
                print(classification_report(self.eval_true_list,
                                            self.eval_output_list,
                                            target_names=self.labels.values(),
                                            zero_division=0))
            except ValueError:
                print('Not defined')
                pass
        if self.confusion:
            c_matrix = confusion_matrix(self.eval_true_list, self.eval_output_list)
            print('Confusion matrix (Evaluation data):')
            print(c_matrix)

        if self.log_probability:
            head = []
            items = []
            probability_arr = np.empty((len(self.log_probability), len(self.log_probability)))
            for key in sorted(self.log_probability):
                head.append(key)
                data = self.log_probability.get(key)
                count = data.get('count')
                p_ = data.get('p_')
                p_ /= count
                p_ = np.round(p_, 3)
                probability_arr[key, :] = p_
                items.append(p_.tolist())
            print()
            print('Probabilities matrix:')
            print(tabulate(items, headers=head), '\n')
            bh_distance = self.__calculate_bh_distance(probability_arr)
            graph = NetGraph(bh_distance, self.labels)
            print('Bhattacharyya_distance:')
            print(tabulate(bh_distance.tolist(), headers=head), '\n')
            kl_distance = self.__calculate_kl_distance(probability_arr)
            print('Kullback Leibler distance:')
            print(tabulate(kl_distance.tolist(), headers=head))

    @staticmethod
    def __calculate_bh_distance(p_array):
        assert p_array is not None
        bh_arr = p_array.copy()
        t_ = bh_arr.transpose(1, 0)
        sim_arr = -np.log(np.sqrt(t_ * p_array))
        sim_arr = 1 * np.ones_like(sim_arr) / sim_arr
        d_arr = np.diag(sim_arr)
        d_arr.setflags(write=True)
        d_arr.fill(0)
        sim_arr = np.round(sim_arr, 3)
        return sim_arr

    @staticmethod
    def __calculate_kl_distance(p_array):
        assert p_array is not None
        kl_arr = p_array.copy()
        t_ = p_array.transpose(1, 0)
        pq = np.log(kl_arr / t_) * kl_arr
        qp = np.log(t_ / kl_arr) * t_
        sim_arr = (pq + qp) / 2
        # o_ = np.ones_like(kl_arr)
        # sim_arr = o_ - sim_arr
        d_arr = np.diag(sim_arr)
        d_arr.setflags(write=True)
        d_arr.fill(0)
        sim_arr = np.round(sim_arr, 4)
        return sim_arr
