from __future__ import division
from __future__ import print_function

import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from models import GCN_1_strati_1, GCN_1_strati_4, GCN_2_strati_1, GCN_2_strati_4, MLP
from utils import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
file = "file_PLI_4_20_0_1c_1s_sf_1"
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')  # 0.01
flags.DEFINE_integer('model_arc', 1,
                     'Architecture of GCN model.')  # 1 = 'GCN 1 strato', 2 = 'GCN 1 strato 4 cluster', 3 = 'GCN 2 strati', 4 = 'GCN 2 strati 4 cluster'
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 512, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('cluster1', 32, 'Number of units in pooling layer 1.')
flags.DEFINE_integer('hidden2', 256, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('cluster2', 16, 'Number of units in pooling layer 2.')
flags.DEFINE_integer('hidden3', 256, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('cluster3', 8, 'Number of units in pooling layer 3.')
flags.DEFINE_integer('hidden4', 512, 'Number units in hidden layer 3.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')  #
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

feat = 'node-label'

# Load data
adj_train, adj_test, features_init, y_train_init, test, y_test, train_mask_init, test_mask = load_data2(
    FLAGS.dataset)

# Some preprocessing
features_init = preprocess_features(features_init)
test = preprocess_features(test)

if FLAGS.model == 'gcn':
    support = preprocess_adj2(adj_train)
    support2 = preprocess_adj2(adj_test)
    num_supports = 1
    if FLAGS.model_arc == 1:
        model_func = GCN_1_strati_1
    elif FLAGS.model_arc == 2:
        model_func = GCN_1_strati_4
    elif FLAGS.model_arc == 3:
        model_func = GCN_2_strati_1
    elif FLAGS.model_arc == 4:
        model_func = GCN_2_strati_4
elif FLAGS.model == 'gcn_cheby':
    fea = list()
    for adj in adj_train:
        fea.append(chebyshev_polynomials(adj, FLAGS.max_degree))
    support = fea
    fea1 = list()
    for adj1 in adj_test:
        fea1.append(chebyshev_polynomials(adj1, FLAGS.max_degree))
    support2 = fea1
    num_supports = 1 + FLAGS.max_degree
    if FLAGS.model_arc == 1:
        model_func = GCN_1_strati_1
    elif FLAGS.model_arc == 2:
        model_func = GCN_1_strati_4
    elif FLAGS.model_arc == 3:
        model_func = GCN_2_strati_1
    elif FLAGS.model_arc == 4:
        model_func = GCN_2_strati_4
elif FLAGS.model == 'dense':
    support = [preprocess_adj2(adj_train)]  # Not used
    support2 = preprocess_adj2(adj_test)  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32, shape=(64, 64)) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_init[0][2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train_init.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features_init[0][2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


def evaluate2(features, support, labels, mask, placeholders):
    t_test = time.time()
    if FLAGS.model == 'gcn_cheby':
        feed_dict_val = construct_feed_dict2(features, support, labels, mask, placeholders)
    else:
        feed_dict_val = construct_feed_dict3(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []


def cross_val(val_idx, data_cross, label_cross, mask_cross):
    val_size = len(data_cross) // 10
    train_graphs = data_cross[:val_idx * val_size]
    train_label = label_cross[:val_idx * val_size, :]
    train_mask = mask_cross[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + data_cross[(val_idx + 1) * val_size:]
        train_label = np.concatenate((train_label, label_cross[(val_idx + 1) * val_size:, :]))
        train_mask = np.concatenate((train_mask, mask_cross[(val_idx + 1) * val_size:]))
    val_graphs = data_cross[val_idx * val_size: (val_idx + 1) * val_size]
    val_label = label_cross[val_idx * val_size: (val_idx + 1) * val_size, :]
    val_mask = mask_cross[val_idx * val_size: (val_idx + 1) * val_size]
    print(val_idx, ' Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs),
          '; Num validation graphs: ', len(train_label))

    return train_graphs, val_graphs, train_label, val_label, train_mask, val_mask


def cross_val2(val_idx, data_cross, label_cross, mask_cross, adj_list):
    val_size = len(data_cross) // 20
    train_graphs = data_cross[:val_idx * val_size]
    train_label = label_cross[:val_idx * val_size, :]
    train_mask = mask_cross[:val_idx * val_size]
    train_adj = adj_list[:val_idx * val_size]
    if val_idx < (val_size - 1):
        train_graphs = train_graphs + data_cross[(val_idx + 1) * val_size:]
        train_label = np.concatenate((train_label, label_cross[(val_idx + 1) * val_size:, :]))
        train_mask = np.concatenate((train_mask, mask_cross[(val_idx + 1) * val_size:]))
        train_adj = train_adj + adj_list[(val_idx + 1) * val_size:]

    val_graphs = data_cross[val_idx * val_size: (val_idx + 1) * val_size]
    val_label = label_cross[val_idx * val_size: (val_idx + 1) * val_size, :]
    val_mask = mask_cross[val_idx * val_size: (val_idx + 1) * val_size]
    val_adj = adj_list[val_idx * val_size: (val_idx + 1) * val_size]
    print(val_idx, ' Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs),
          '; Num label graphs: ', len(train_label))

    return train_graphs, val_graphs, train_label, val_label, train_mask, val_mask, train_adj, val_adj


loss_t = []
acc_t = []
loss_v = []
acc_v = []

for cross in range(1):
    data_cross = features_init
    label_cross = y_train_init
    mask_cross = train_mask_init
    # mask is resilient and it is ones vector
    features, val, y_train, y_val, train_mask, val_mask, adj_train_support, adj_val_support = \
        cross_val2(cross, data_cross, label_cross, mask_cross, support)
    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        outs = []
        acc_tmp = list()
        loss_tmp_t = list()
        for f in tqdm(range(len(features))):
            if FLAGS.model == 'gcn_cheby':
                feed_dict = construct_feed_dict2(features[f], support[f], np.reshape(y_train[f], [1, 4]),
                                                 train_mask[f], placeholders)
            else:
                feed_dict = construct_feed_dict3(features[f], support[f], np.reshape(y_train[f], [1, 4]),
                                                 train_mask[f], placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            acc_med = 0
            loss_med = 0
            outs = sess.run([model.opt_op, model.loss, model.accuracy,
                             model.activations[0]],
                            feed_dict=feed_dict)
            loss_tmp_t.append(outs[1])
            acc_tmp.append(outs[2])

        acc_med = np.mean(np.array(acc_tmp))
        acc_t.append(acc_med)
        loss_med = np.mean(np.array(loss_tmp_t))
        loss_t.append(loss_med)
        num_acc = np.where(np.array(acc_tmp) == 1)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), " train_loss=", "{:.5f}".format(loss_med),
              " train_acc=", "{:.5f}".format(acc_med),
              "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")

plt.figure(1)
plt.plot(range(0, len(loss_t) * 2, 2), loss_t, 'r--', label='Loss T')
plt.legend(loc='upper right')
plt.title('loss training')
plt.savefig(file + '_loss.png', format='png', bbox_inches='tight', transparent=False, dpi=600)
# plt.show()

plt.figure(2)
plt.plot(range(0, len(acc_t) * 2, 2), acc_t, 'b--', label='Accuracy T')
plt.legend(loc='upper left')
plt.title('accuracy training vs accuracy validation')
plt.savefig(file + '_accuracy.png', format='png', bbox_inches='tight', transparent=False, dpi=600)
# plt.show()

acc_tmp2 = list()
duration_tmp2 = list()
res_list = list()
test_cost = 0
# Testing
for p in tqdm(range(len(test))):
    test_cost, test_acc, test_duration, res = evaluate2(test[p], support2[p], np.reshape(y_test[p], [1, 4]),
                                                        test_mask[p], placeholders)
    acc_tmp2.append(test_acc)
    duration_tmp2.append(test_duration)
    res_list.append(res)

with open(file + "_y.txt", "w") as f:
    for s in y_test:
        f.write(str(s) + "\n")

with open(file + "_res.txt", "w") as f:
    for s in res_list:
        f.write(str(s) + "\n")

test_acc2 = np.mean(np.array(acc_tmp2))
test_duration2 = np.mean(np.array(duration_tmp2))
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc2), "time=", "{:.5f}".format(test_duration2))
