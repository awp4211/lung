"""
Utility
plot training acc and test acc
plot training cost
"""


def plot_acc(train_accuracy,
             val_accuracy,
             n_epoch,
             name,
             train_best,
             val_best
             ):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.xlabel('n_epoch')
    plt.ylabel('accuracy')

    x = range(n_epoch)

    if train_accuracy:
        plt.plot(x, train_accuracy, "-", label="train_accuracy")
    if val_accuracy:
        plt.plot(x, val_accuracy, "-", label="val_accuracy")

    plt.grid(True)
    plt.title("{0}_train_best:{1}_val_best:{2}".format(name, train_best, val_best))
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc=1, borderaxespad=0.)

    plt.savefig('result/{0}.png'.format(name), dpi=400, bbox_inches='tight')


def plot_cost(costs,
              n_epoch,
              name
              ):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.xlabel('n_epoch')
    plt.ylabel('cost')

    x = range(n_epoch)
    plt.plot(x, costs, "-")

    plt.grid(True)
    plt.title(name)
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc=1, borderaxespad=0.)

    plt.savefig('result/{0}.png'.format(name), dpi=400, bbox_inches='tight')