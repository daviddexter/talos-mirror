def test_rest(scan_object):

    print('\n >>> start testing the rest... \n')

    import talos

    import random

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    

    from tensorflow.keras import metrics , losses

    deploy_filename = 'test' + str(random.randint(1, 20000000000))

    print('\n ...Deploy()... \n')
    talos.Deploy(scan_object, deploy_filename, 'val_acc')

    print('\n ...Restore()... \n')
    restored = talos.Restore(deploy_filename + '.zip')

    x, y = talos.templates.datasets.breast_cancer()
    x = x[:50]
    y = y[:50]

    x_train, y_train, x_val, y_val = talos.utils.val_split(x, y, .2)
    x = talos.utils.rescale_meanzero(x)

    callbacks = [
                 talos.utils.early_stopper(10),
                 talos.utils.ExperimentLogCallback('test', {})]

    metrics = [metrics.MeanAbsolutePercentageError(),
               metrics.MeanSquaredLogarithmicError(),
               metrics.RootMeanSquaredError(),
               metrics.Precision(),
               metrics.Recall()]


    print('\n ...callbacks and metrics... \n')

    model1 = Sequential()
    model1.add(Dense(10, input_dim=x.shape[1]))
    model1.add(Dense(1))
    model1.compile('adam', metrics=metrics)
    model1.fit(x, y, callbacks=callbacks)

    print('\n ...generator... \n')

    model2 = Sequential()
    model2.add(Dense(10, input_dim=x.shape[1]))
    model2.add(Dense(1))
    model2.compile('adam')
    model2.fit_generator(talos.utils.generator(x, y, 10), 5)

    print('\n ...SequenceGenerator... \n')

    model3 = Sequential()
    model3.add(Dense(10, input_dim=x.shape[1]))
    model3.add(Dense(1))
    model3.compile('adam', 'logcosh')
    model3.fit_generator(talos.utils.SequenceGenerator(x, y, 10))

    # print('\n ...gpu_utils... \n')

    # talos.utils.gpu_utils.force_cpu()
    # talos.utils.gpu_utils.parallel_gpu_jobs()

    # print('\n ...gpu_utils... \n')

    from talos.utils.test_utils import create_param_space
    create_param_space(restored.results, 5)

    print('finished testing the rest \n')
