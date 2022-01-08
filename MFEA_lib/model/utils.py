import pickle
import matplotlib.pyplot as plt
import numpy as np
from ..tasks.function import AbstractFunc
from ..model.mfea import AbstractModel

def saveModel(model, PATH: str):
    '''
    `.mso`
    '''
    f = open(PATH, 'wb')
    pickle.dump(model, f)
    f.close()
    return 'Saved'
def loadModel(PATH: str) -> AbstractModel:
    '''
    `.mso`
    '''
    f = open(PATH, 'rb')
    model = pickle.load(f)
    f.close()
    return model

def compareModel(models: list, tasks: list[AbstractFunc], shape:tuple = None, label_legend = None,
    upper_generation = None, step = 1, figsize = (30, 10), dpi = 200, yscale:str = None):
    fig = plt.figure(figsize = figsize, dpi= dpi)
    fig.suptitle("Compare Models\n", size = 15)
    fig.set_facecolor("white")

    if label_legend is None:
        label_legend = [t.__class__.__name__ for t in models]
    else:
        assert len(models) == len(label_legend)

    if shape is None:
        shape = (1, len(tasks))
    else:
        assert shape[0] * shape[1] == len(tasks)

    if upper_generation is None:
        upper_generation = min([len(m.history_cost) for m in models])
    else:
        upper_generation = min(upper_generation + 1, min([len(m.history_cost) for m in models]) + 1)

    for idx_t, task in enumerate(tasks):
        for idx_m, model in enumerate(models):
            plt.subplot(shape[0], shape[1], idx_t + 1)

            plt.plot(np.arange(0, upper_generation, step), 
                        model.history_cost[np.arange(0, upper_generation, step), idx_t],
                    label = label_legend[idx_m]
            )
            if step != 1:
                plt.scatter(np.arange(0, upper_generation, step), 
                        model.history_cost[np.arange(0, upper_generation, step), idx_t]
                )
            plt.legend()
        
        plt.title(task.name)
        plt.xlabel("Generations")
        plt.ylabel("History Cost")
        if yscale is not None:
            plt.yscale(yscale)

    plt.show()
    return fig