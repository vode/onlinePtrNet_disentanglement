# Online Pointer Network for Conversation Disentanglement

This is the repo for our EMNLP2020 paper Online Pointer Network for Conversation Disentanglement

The ***model.py*** contains our model, ***JointModel*** class contains the network for both *reply-to* relationship and *same conversation* relationship

The ***disentangle.py*** is the main script to train and evaluate the model. I have upload our best model file under data folder.

> bash eval.sh

To run the trained model.

For the self-link re-threshold, please the ***decoding function*** in ***disentangle.py***