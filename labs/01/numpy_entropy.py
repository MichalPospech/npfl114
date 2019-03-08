#!/usr/bin/env python3
import numpy as np
def cross_ent(x,y):
    return -np.sum(np.log(y)*x)


if __name__ == "__main__":
    # Load data distribution, each data point on a line
    dist_data = []
    data_values = set()
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            data_values.add(line)
            dist_data.append(line)


   


    # Load model distribution, each line `word \t probability`.
    probs = dict()
    for value in data_values:
        probs[value] = 0
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            tab_index =  line.find("\t")
            value=line[:tab_index]
            data_values.add(value)
            probs[value]=float(line[tab_index+1:])

    
    model_nparr = np.array([probs[x] for x in data_values])
    data_nparr = np.array([dist_data.count(x)/len(dist_data) for x in data_values])
    entropy = cross_ent(data_nparr[np.nonzero(data_nparr)],data_nparr[np.nonzero(data_nparr)])
    cross_entropy = cross_ent(data_nparr,model_nparr)
    kl_divergence = cross_entropy - entropy
   
    print("{:.2f}".format(entropy))
    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(kl_divergence))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
