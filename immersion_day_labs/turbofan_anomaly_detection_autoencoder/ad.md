# Lab B: Predictive Maintenance with AutoEncoders

In this lab, we'll work our way through an example Jupyter notebook that demonstrates how to use a custom autoencoder neural network.  This labs will assume that we only have unlabled data from the engine operations.  Many times you won't have a labeled dataset to train a classification model.  

To convert to problem from an unsupervised problem to supervised, we will use an autoencoder model.  An autoencoder is a neural network model where the input matches the output.  The idea is to recreate the input data with the network.  The network is trained on normal aircraft behavior.  Once trained, new examples are fed to the network.  The reconstruction error is used as the metric to identify anomalies in the dataset.   This approach is very powerful and has been applied to computer vision, time series, NLP problems. 

![AutoEncoder](img/autoencoder.png)

To begin, follow these steps:

1. [Click this link](notebooks/TurbofanRUL_AutoEncoder.ipynb) to download the TurbofanRUL_AutoEncoder.ipynb Jupyter notebook to your local machine.
2. [Click this link](notebooks/turbofan_autoencoder_keras_tf.py) to download the turbofan_autoencoder_keras_tf.py script to your local machine.
2. In your notebook instance, click the **New** button on the right and select **Folder**.  
3. Click the checkbox next to your new folder, click the **Rename** button above in the menu bar, and give the folder a name such as 'notebooks'.
4. Click the folder to enter it.
5. To upload the notebook, click the **Upload** button on the right, then in the file selection popup, select the file '**TurbofanRUL_AutoEncoder.ipynb**' from the folder on your computer where you downloaded the file. Then click the blue **Upload** button that appears in the notebook next to the file name.
6. You are now ready to begin the notebook:  click the notebook's file name to open it.

!!! Done
    Great! You have just run a  Jupyter Notebook in SageMaker.
	

