#!/usr/bin/env/python
from __future__ import print_function
from typing import List, Any, Sequence
from utils import MLP, ThreadedIterator

import tensorflow as tf
import time
import os
import json
import numpy as np
import pickle
import random
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，以支持在无图形界面的环境中运行
import matplotlib.pyplot as plt

class DetectModel(object):
    @classmethod
    def default_params(cls):
        return {
            'num_epochs': 250,
            'patience': 200,
            'learning_rate': 0.002,
            'clamp_gradient_norm': 0.9,    # [0.8, 1.0]
            'out_layer_dropout_keep_prob': 0.9,    # [0.8, 1.0]

            'hidden_size': 256, # 256/512/1024/2048
            'use_graph': True,

            'tie_fwd_bkwd': False,  # True or False
            'task_ids': [0],

            # 'train_file': 'train_data/reentrancy/train.json',
            # 'valid_file': 'train_data/reentrancy/valid.json'

            'train_file': 'train_data/timestamp/train.json',
            'valid_file': 'train_data/timestamp/valid.json'
        }

    def __init__(self, args):
        self.args = args

        data_dir = ''
        if '--data_dir' in args and args['--data_dir'] is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        random_seed = args.get('--random_seed')
        self.random_seed = int(9903)  # optional

        threshold = args.get('--thresholds')
        self.threshold = float(0.45)  # optional

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
        self.best_model_file = os.path.join(log_dir, "%s_model_best.pickle" % self.run_id)

        params = self.default_params()
        config_file = args.get('--config-file')
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        config = args.get('--config')
        if config is not None:
            params.update(json.loads(config))
        self.params = params

        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        print("Run with current seed %s " % self.random_seed)

        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.num_graph = 1
        self.train_num_graph = 0
        self.valid_num_graph = 0

        self.train_data, self.train_num_graph = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data, self.valid_num_graph = self.load_data(params['valid_file'], is_training_data=False)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            self.placeholders = {}
            self.weights = {}
            self.ops = {}
            self.make_model()
            self.make_train_step()

            restore_file = args.get('--restore')
            if restore_file is not None:
                self.restore_model(restore_file)
            else:
                self.initialize_model()

        # 训练和验证数据的收集列表
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.train_precisions = []
        self.valid_precisions = []
        self.train_recalls = []
        self.valid_recalls = []
        self.train_f1_scores = []
        self.valid_f1_scores = []
        self.valid_aucs = []

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading baseline from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")
        if restrict is not None and restrict > 0:
            data = data[:restrict]

        num_fwd_edge_types = 0
        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph'] for v in [e[0], e[2]]]))
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features"][0]))

        return self.process_raw_graphs(data, is_training_data)

    @staticmethod
    def graph_string_to_array(graph_string: str) -> List[List[int]]:
        return [[int(v) for v in s.split(' ')]
                for s in graph_string.split('\n')]

    def process_raw_graphs(self, raw_data: Sequence[Any], is_training_data: bool) -> Any:
        raise Exception("Models have to implement process_raw_graphs!")

    def make_model(self):
        self.placeholders['target_values'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                            name='target_values')
        self.placeholders['target_mask'] = tf.placeholder(tf.float32, [len(self.params['task_ids']), None],
                                                          name='target_mask')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, [], name='num_graphs')

        self.placeholders['out_layer_dropout_keep_prob'] = tf.placeholder(tf.float32, [],
                                                                          name='out_layer_dropout_keep_prob')

        with tf.variable_scope("graph_model"):
            self.prepare_specific_graph_model()
            if self.params['use_graph']:
                self.ops['final_node_representations'] = self.compute_final_node_representations()
            else:
                self.ops['final_node_representations'] = tf.zeros_like(self.placeholders['process_raw_graphs'])

        self.ops['losses'] = []
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    self.weights['regression_gate_task%i' % task_id] = MLP(2 * self.params['hidden_size'], 1, [],
                                                                           self.placeholders[
                                                                               'out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    self.weights['regression_transform_task%i' % task_id] = MLP(self.params['hidden_size'], 1, [],
                                                                                self.placeholders[
                                                                                    'out_layer_dropout_keep_prob'])
                computed_values, sigm_val, initial_re = self.gated_regression(self.ops['final_node_representations'],
                                                                              self.weights[
                                                                                  'regression_gate_task%i' % task_id],
                                                                              self.weights[
                                                                                  'regression_transform_task%i' % task_id])

                def f(x):
                    x = 1 * x
                    x = x.astype(np.float32)
                    return x

                new_computed_values = tf.nn.sigmoid(computed_values)
                new_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=computed_values,
                                                                                  labels=self.placeholders[
                                                                                             'target_values'][internal_id, :]))
                a = tf.math.greater_equal(new_computed_values, self.threshold)
                a = tf.py_func(f, [a], tf.float32)
                correct_pred = tf.equal(a, self.placeholders['target_values'][internal_id, :])
                self.ops['new_computed_values'] = new_computed_values
                self.ops['sigm_val'] = sigm_val
                self.ops['initial_re'] = initial_re
                self.ops['accuracy_task%i' % task_id] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                b = tf.multiply(self.placeholders['target_values'][internal_id, :], 2)
                b = tf.py_func(f, [b], tf.float32)
                c = tf.cast(a, tf.float32)
                d = tf.math.add(b, c)
                self.ops['sigm_c'] = correct_pred

                d_TP = tf.math.equal(d, 3)
                TP = tf.reduce_sum(tf.cast(d_TP, tf.float32))
                d_FN = tf.math.equal(d, 2)
                FN = tf.reduce_sum(tf.cast(d_FN, tf.float32))
                d_FP = tf.math.equal(d, 1)
                FP = tf.reduce_sum(tf.cast(d_FP, tf.float32))
                d_TN = tf.math.equal(d, 0)
                TN = tf.reduce_sum(tf.cast(d_TN, tf.float32))
                self.ops['sigm_sum'] = tf.add_n([TP, FN, FP, TN])
                self.ops['sigm_TP'] = TP
                self.ops['sigm_FN'] = FN
                self.ops['sigm_FP'] = FP
                self.ops['sigm_TN'] = TN

                R = tf.cast(tf.divide(TP, tf.add(TP, FN)), tf.float32)
                P = tf.cast(tf.divide(TP, tf.add(TP, FP)), tf.float32)
                FPR = tf.cast(tf.divide(FP, tf.add(TN, FP)), tf.float32)
                D_TP = tf.add(TP, TP)
                F1 = tf.cast(tf.divide(D_TP, tf.add_n([D_TP, FP, FN])), tf.float32)
                self.ops['sigm_Recall'] = R
                self.ops['sigm_Precision'] = P
                self.ops['sigm_F1'] = F1
                self.ops['sigm_FPR'] = FPR
                self.ops['losses'].append(new_loss)
        self.ops['loss'] = tf.reduce_sum(self.ops['losses'])

    def make_train_step(self):
        trainable_vars = self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if self.args.get('--freeze-graph-model'):
            graph_vars = set(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
            filtered_vars = []
            for var in trainable_vars:
                if var not in graph_vars:
                    filtered_vars.append(var)
                else:
                    print("Freezing weights of variable %s." % var.name)
            trainable_vars = filtered_vars
        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        grads_and_vars = optimizer.compute_gradients(self.ops['loss'], var_list=trainable_vars)
        clipped_grads = []
        for grad, var in grads_and_vars:
            if grad is not None:
                clipped_grads.append((tf.clip_by_norm(grad, self.params['clamp_gradient_norm']), var))
            else:
                clipped_grads.append((grad, var))
        self.ops['train_step'] = optimizer.apply_gradients(clipped_grads)
        self.sess.run(tf.local_variables_initializer())

    def gated_regression(self, last_h, regression_gate, regression_transform):
        raise Exception("Models have to implement gated_regression!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self) -> tf.Tensor:
        raise Exception("Models have to implement compute_final_node_representations!")

    def make_minibatch_iterator(self, data: Any, is_training: bool):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def run_epoch(self, epoch_name: str, data, epoch, is_training: bool):
        chemical_accuracies = np.array([0.066513725, 0.012235489, 0.071939046, 0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926, 0.00409976, 0.004527465, 0.012292586,
                                        0.037467458])

        loss = 0
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        tps = []
        fns = []
        fps = []
        tns = []
        start_time = time.time()
        processed_graphs = 0
        accuracy_ops = [self.ops['accuracy_task%i' % task_id] for task_id in self.params['task_ids']]
        precision_ops = [self.ops['sigm_Precision']]
        recall_ops = [self.ops['sigm_Recall']]
        f1_ops = [self.ops['sigm_F1']]
        tp_ops = [self.ops['sigm_TP']]
        fn_ops = [self.ops['sigm_FN']]
        fp_ops = [self.ops['sigm_FP']]
        tn_ops = [self.ops['sigm_TN']]
        all_preds = []
        all_labels = []

        batch_iterator = ThreadedIterator(self.make_minibatch_iterator(data, is_training), max_queue_size=5)
        for step, batch_data in enumerate(batch_iterator):
            num_graphs = batch_data[self.placeholders['num_graphs']]
            processed_graphs += num_graphs
            if is_training:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = self.params['out_layer_dropout_keep_prob']
                fetch_list = [self.ops['loss']] + accuracy_ops + precision_ops + recall_ops + f1_ops + tp_ops + fn_ops + fp_ops + tn_ops + [self.ops['train_step'], self.ops['new_computed_values']]
            else:
                batch_data[self.placeholders['out_layer_dropout_keep_prob']] = 1.0
                fetch_list = [self.ops['loss']] + accuracy_ops + precision_ops + recall_ops + f1_ops + tp_ops + fn_ops + fp_ops + tn_ops + [self.ops['new_computed_values']]

            result = self.sess.run(fetch_list, feed_dict=batch_data)
            if is_training:
                batch_loss = result[0]
                batch_accuracies = result[1:1+len(accuracy_ops)]
                batch_precisions = result[1+len(accuracy_ops):1+len(accuracy_ops)+len(precision_ops)]
                batch_recalls = result[1+len(accuracy_ops)+len(precision_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)]
                batch_f1_scores = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)]
                batch_tps = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)]
                batch_fns = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops)]
                batch_fps = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops)+len(fp_ops)]
                batch_tns = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops)+len(fp_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops)+len(fp_ops)+len(tn_ops)]
                _ = result[-2]
                preds = result[-1]
            else:
                batch_loss = result[0]
                batch_accuracies = result[1:1+len(accuracy_ops)]
                batch_precisions = result[1+len(accuracy_ops):1+len(accuracy_ops)+len(precision_ops)]
                batch_recalls = result[1+len(accuracy_ops)+len(precision_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)]
                batch_f1_scores = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)]
                batch_tps = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)]
                batch_fns = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops)]
                batch_fps = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops)+len(fp_ops)]
                batch_tns = result[1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops)+len(fp_ops):1+len(accuracy_ops)+len(precision_ops)+len(recall_ops)+len(f1_ops)+len(tp_ops)+len(fn_ops)+len(fp_ops)+len(tn_ops)]
                preds = result[-1]

            labels = batch_data[self.placeholders['target_values']][0, :]
            all_preds.extend(preds)
            all_labels.extend(labels)

            loss += batch_loss * num_graphs
            accuracies.append(np.array(batch_accuracies) * num_graphs)
            precisions.append(np.array(batch_precisions) * num_graphs)
            recalls.append(np.array(batch_recalls) * num_graphs)
            f1_scores.append(np.array(batch_f1_scores) * num_graphs)
            tps.append(np.array(batch_tps) * num_graphs)
            fns.append(np.array(batch_fns) * num_graphs)
            fps.append(np.array(batch_fps) * num_graphs)
            tns.append(np.array(batch_tns) * num_graphs)

            print("Running %s, batch %i (has %i graphs). "
                  "Loss so far: %.4f" % (epoch_name, step, num_graphs, loss / processed_graphs), end='\r')

        accuracies = np.sum(accuracies, axis=0) / processed_graphs
        loss = loss / processed_graphs
        precisions = np.sum(precisions, axis=0) / processed_graphs
        recalls = np.sum(recalls, axis=0) / processed_graphs
        f1_scores = np.sum(f1_scores, axis=0) / processed_graphs
        tps = np.sum(tps, axis=0) / processed_graphs
        fns = np.sum(fns, axis=0) / processed_graphs
        fps = np.sum(fps, axis=0) / processed_graphs
        tns = np.sum(tns, axis=0) / processed_graphs
        error_ratios = accuracies / chemical_accuracies[self.params["task_ids"]]
        instance_per_sec = processed_graphs / (time.time() - start_time)

        if not is_training:
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            try:
                auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                auc = float('nan')
        else:
            auc = float('nan')

        return loss, accuracies[0], precisions[0], recalls[0], f1_scores[0], tps[0], fns[0], fps[0], tns[0], auc, instance_per_sec

    def train(self):
        train_losses = []
        train_accuracies = []
        train_precisions = []
        train_recalls = []
        train_f1_scores = []
        train_tps = []
        train_fns = []
        train_fps = []
        train_tns = []
        valid_losses = []
        valid_accuracies = []
        valid_precisions = []
        valid_recalls = []
        valid_f1_scores = []
        valid_tps = []
        valid_fns = []
        valid_fps = []
        valid_tns = []
        valid_aucs = []
        log_to_save = []
        total_time_start = time.time()
        with self.graph.as_default():
            if self.args.get('--restore') is not None:
                _, valid_accs, _, _, _, _, _, _, _, _, _ = self.run_epoch("Resumed (validation)", self.valid_data, 0, False)
                best_val_acc = np.sum(valid_accs)
                best_val_acc_epoch = 0
                print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" % best_val_acc)
            else:
                (best_val_acc, best_val_acc_epoch) = (float("+inf"), 0)
            for epoch in range(1, self.params['num_epochs'] + 1):
                print("== Epoch %i" % epoch)
                train_start = time.time()
                self.num_graph = self.train_num_graph
                train_loss, train_acc, train_precision, train_recall, train_f1, train_tp, train_fn, train_fp, train_tn, _, train_speed = self.run_epoch("epoch %i (training)" % epoch, self.train_data, epoch, True)
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)
                train_precisions.append(train_precision)
                train_recalls.append(train_recall)
                train_f1_scores.append(train_f1)
                train_tps.append(train_tp)
                train_fns.append(train_fn)
                train_fps.append(train_fp)
                train_tns.append(train_tn)
                print("\r\x1b[K Train: loss: %.5f | acc: %.5f | precision: %.5f | recall: %.5f | f1: %.5f | TP: %.2f | FN: %.2f | FP: %.2f | TN: %.2f | instances/sec: %.2f" % (
                    train_loss, train_acc, train_precision, train_recall, train_f1, train_tp, train_fn, train_fp, train_tn, train_speed))
                epoch_time_train = time.time() - train_start
                print(epoch_time_train)

                valid_start = time.time()
                self.num_graph = self.valid_num_graph
                valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_tp, valid_fn, valid_fp, valid_tn, valid_auc, valid_speed = self.run_epoch("epoch %i (validation)" % epoch, self.valid_data, epoch, False)
                valid_losses.append(valid_loss)
                valid_accuracies.append(valid_acc)
                valid_precisions.append(valid_precision)
                valid_recalls.append(valid_recall)
                valid_f1_scores.append(valid_f1)
                valid_tps.append(valid_tp)
                valid_fns.append(valid_fn)
                valid_fps.append(valid_fp)
                valid_tns.append(valid_tn)
                valid_aucs.append(valid_auc)
                print("\r\x1b[K Valid: loss: %.5f | acc: %.5f | precision: %.5f | recall: %.5f | f1: %.5f | TP: %.2f | FN: %.2f | FP: %.2f | TN: %.2f | auc: %.5f | instances/sec: %.2f" % (
                    valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_tp, valid_fn, valid_fp, valid_tn, valid_auc, valid_speed))
                epoch_time_valid = time.time() - valid_start
                print(epoch_time_valid)

                epoch_time_total = time.time() - total_time_start
                print(epoch_time_total)
                log_entry = {
                    'epoch': epoch,
                    'time': epoch_time_total,
                    'train_results': (train_loss, train_acc, train_precision, train_recall, train_f1, train_tp, train_fn, train_fp, train_tn, train_speed),
                    'valid_results': (valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_tp, valid_fn, valid_fp, valid_tn, valid_auc, valid_speed),
                }
                log_to_save.append(log_entry)

            # 训练完成后绘制并保存图表
            self.plot_results(train_losses, valid_losses, train_accuracies, valid_accuracies,
                              train_precisions, valid_precisions, train_recalls, valid_recalls,
                              train_f1_scores, valid_f1_scores, valid_aucs)

    def plot_results(self, train_losses, valid_losses, train_accuracies, valid_accuracies,
                     train_precisions, valid_precisions, train_recalls, valid_recalls,
                     train_f1_scores, valid_f1_scores, valid_aucs):
        epochs = range(1, len(train_losses) + 1)

        # 创建一个 2x3 的子图布局
        plt.figure(figsize=(18, 10))

        # 子图1：训练和验证 Loss
        plt.subplot(2, 3, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, valid_losses, 'r-', label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # 子图2：训练和验证 Accuracy
        plt.subplot(2, 3, 2)
        plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
        plt.plot(epochs, valid_accuracies, 'r-', label='Valid Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        # 子图3：训练和验证 Precision
        plt.subplot(2, 3, 3)
        plt.plot(epochs, train_precisions, 'b-', label='Train Precision')
        plt.plot(epochs, valid_precisions, 'r-', label='Valid Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Precision over Epochs')
        plt.legend()

        # 子图4：训练和验证 Recall
        plt.subplot(2, 3, 4)
        plt.plot(epochs, train_recalls, 'b-', label='Train Recall')
        plt.plot(epochs, valid_recalls, 'r-', label='Valid Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Recall over Epochs')
        plt.legend()

        # 子图5：训练和验证 F1 Score
        plt.subplot(2, 3, 5)
        plt.plot(epochs, train_f1_scores, 'b-', label='Train F1 Score')
        plt.plot(epochs, valid_f1_scores, 'r-', label='Valid F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score over Epochs')
        plt.legend()

        # 子图6：验证 AUC
        plt.subplot(2, 3, 6)
        plt.plot(epochs, valid_aucs, 'r-', label='Valid AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC over Epochs')
        plt.legend()

        # 保存图表到文件
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.close()  # 关闭图表以释放内存

    def save_model(self, path: str) -> None:
        weights_to_save = {}
        for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            assert variable.name not in weights_to_save
            weights_to_save[variable.name] = self.sess.run(variable)

        data_to_save = {
            "params": self.params,
            "weights": weights_to_save
        }

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def initialize_model(self) -> None:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def restore_model(self, path: str) -> None:
        print("Restoring weights from file %s." % path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        assert len(self.params) == len(data_to_load['params'])
        for (par, par_value) in self.params.items():
            if par not in ['task_ids', 'num_epochs']:
                assert par_value == data_to_load['params'][par]

        variables_to_initialize = []
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                used_vars.add(variable.name)
                if variable.name in data_to_load['weights']:
                    restore_ops.append(variable.assign(data_to_load['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in data_to_load['weights']:
                if var_name not in used_vars:
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            self.sess.run(restore_ops)