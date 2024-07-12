import cPickle


def loadSiteSeer():
    DATASET = 'citeseer'
    NAMES = ['x', 'y', 'tx', 'ty', 'graph']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(cPickle.load(open("data/trans.{}.{}".format(DATASET, NAMES[i]))))

    # load the data: x, y, tx, ty, graph
    trans = tuple(OBJECTS)
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'graph']
    OBJECTS = []
    for i in range(len(NAMES)):
        OBJECTS.append(cPickle.load(open("data/ind.{}.{}".format(DATASET, NAMES[i]))))
    
    # load the data: x, y, tx, ty, allx, graph
    ind = tuple(OBJECTS)

    return trans, ind

