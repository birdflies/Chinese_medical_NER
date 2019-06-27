# encoding=utf-8

"""
bert-blstm-crf layer
@Author:Macan
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf


class BLSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, max_seq_length, labels, seq_lengths, is_training):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param max_seq_length: 序列最大长度
        :param labels: 真实标签
        :param seq_lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.max_seq_length = max_seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.seq_lengths = seq_lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training

    def add_blstm_crf_layer(self, crf_only):
        """
        blstm-crf网络
        :return:
        """
        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            logits = self.bilstm_layer(self.embedded_chars)

        # crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.seq_lengths)
        return (loss, logits, trans, pred_ids)

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw, cell_bw

    def bilstm_layer(self, embedding_chars):
        with tf.variable_scope('bilstm'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=embedding_chars,
                sequence_length=self.seq_lengths,
                dtype=tf.float32)
            lstm_output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            lstm_output = tf.nn.dropout(lstm_output, self.dropout_rate)

        with tf.variable_scope('proj'):
            W = tf.get_variable(name="W",
                                shape=[self.hidden_unit * 2, self.num_labels],
                                dtype=tf.float32,
                                initializer=self.initializers.xavier_initializer())

            b = tf.get_variable(name="b",
                                shape=[self.num_labels],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            output = tf.reshape(lstm_output, shape=[-1, self.hidden_unit * 2])
            pred = tf.matmul(output, W) + b

            logits = tf.reshape(pred, [-1], self.max_seq_length, self.num_labels)

            return logits

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            log_likelihood, transition_params = crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels,
                sequence_lengths=self.seq_lengths)
            return tf.reduce_mean(-log_likelihood), transition_params

    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            W = tf.get_variable("W",
                                shape=[self.embedding_dims, self.num_labels],
                                dtype=tf.float32,
                                initializer=self.initializers.xavier_initializer())

            b = tf.get_variable("b",
                                shape=[self.num_labels],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            output = tf.reshape(self.embedded_chars, shape=[-1, self.embedding_dims])
            pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            logits = tf.reshape(pred, [-1, self.max_seq_length, self.num_labels])

            return logits
