import tensorflow as tf


from step2_train_model1 import train_c3d as model_1
from step2_train_model2 import train_c3d as model_2


if __name__ == "__main__":

    count = 0
    # grid search model1
    for learning_rate in [1e-4, 1e-6]:
        for dropout_keep_prob in [0.5, 0.8]:
            tf.reset_default_graph()

            train_accs, train_costs, train_f1s, val_accs, val_costs, val_f1s, best_train_f1, best_val_f1 =\
                model_1(dropout_keep_prob=dropout_keep_prob,
                        edge_size=48,
                        learning_rate=learning_rate,
                        n_epochs=20,
                        batch_size=30,
                        n_classes=2,
                        show_batch_result=False)


    # grid search model2
    for learning_rate in [1e-4, 1e-6]:
        for dropout_keep_prob in [0.5, 0.8]:
            tf.reset_default_graph()

            train_accs, train_costs, train_f1s, val_accs, val_costs, val_f1s, best_train_f1, best_val_f1 = \
                model_2(dropout_keep_prob=dropout_keep_prob,
                        edge_size=48,
                        learning_rate=learning_rate,
                        n_epochs=20,
                        batch_size=15,
                        n_classes=2,
                        show_batch_result=False)
