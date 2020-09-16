# MLHEP 2020: Variational sparsification for PID classification

In this challenge, we are asking you to train a classifier to identify the type of a particle. There are six particle types: electron, proton, muon, kaon, pion and ghost. Ghost is a particle with another type than the first five or detector noise.

Your task is to achieve a high quality of classification but also sparsify your network as much as possible. You are provided with the LinearSVDO-layer realization that you should use in this challenge.

In order to submit your solution please follow the instructions below:

1. Register in [CodaLab](https://codalab.coresearch.club/), with the same email you have on github.

2. Fork baseline(this one) repository.

3. Play with the architecture of the network in sparse_model.py and play with the optimization routine in sparse_particle_identification.ipynb to improve your score.

4. Create zip-archive with weights of your network(model_weights.pt) and model architecture(sparse_model.py).

5. Submit your zip-archive into the [competititon](https://codalab.coresearch.club/competitions/114)
