Models
========


Introduction
------------------


Models is the result of the Thesis of FratosVR. In this repository you can find the code and documentation of the project.

This includes the code to train and use models to detect gestures made by a person using a mocap suit. This are simple models to ensure easy and fast computation.

Installation
------------------

To install the packages needed to run and train the models, you can use the following command:

```bash
pip install -r requirements.txt
```

Usage
---------

Step by step guide on how to train the models:

 1. **Download a valid dataset**: ensure that the dataset is from a mocap suit and contains the necessary data for training.

 2. **Launch the interface**: run ```python Interface.py``` to open the web interface.

 3. **Select the dataset**: in the interface, choose the dataset you downloaded in step 1.

 4. **Select the model**: choose the model you want to train from the available options.

 5. **Train the model**: click on the "Train" button to start the training process. The interface will show the progress and results of the training.

 6. **Evaluate the model**: after training, you can evaluate the model's performance using the provided metrics in the interface.



You can add new models by creating a new class with the same structure as the existing ones. Make sure to implement the necessary methods and follow the naming conventions used in the project.


Modules
--------

The modules can be separated into two categories: the ones that are used to train models, and the ones that are models:

Trainers
^^^^^^^^^^^

   - :doc:`autoapi/src/DataLoader/index`
   - :doc:`autoapi/src/Interface/index`
   - :doc:`autoapi/src/Utils/index`

Models
^^^^^^^^^^

   - :doc:`autoapi/src/LSTMTrainer/index`
   - :doc:`autoapi/src/CNNTrainer/index`
   - :doc:`autoapi/src/RNNTrainer/index`
   - :doc:`autoapi/src/RandomForestTrainer/index`

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   autoapi/src/DataLoader/index
   autoapi/src/Interface/index
   autoapi/src/Utils/index
   autoapi/src/LSTMTrainer/index
   autoapi/src/CNNTrainer/index
   autoapi/src/RNNTrainer/index
   autoapi/src/RandomForestTrainer/index
