Welcome to tensorboardX's documentation!
========================================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   tensorboard
   utils
   tutorial
   tutorial_zh


What is tensorboardX?
---------------------

Google’s tensorflow’s tensorboard is a web server to serve visualizations of the training
progress of a neural network, it visualizes scalar values, images, text, etc.; these
information are saved as events in tensorflow. It’s a pity that other deep learning
frameworks lack of such tool, so there are already packages letting users to log the
events without tensorflow; however they only provides basic functionalities. The purpose
of this package is to let researchers use a simple interface to log events within PyTorch
(and then show visualization in tensorboard). This package currently supports logging
scalar, image, audio, histogram, text, embedding, and the route of back-propagation. The
following manual is tested on Ubuntu and Mac, and the environment are anaconda’s python2
and python3.

.. admonition:: Naming conflict...

   At first, the package was named tensorboard, and soon there were issues about name
   confliction. The first alternative name came to my mind is tensorboard-pytorch, but in
   order to make it more general, I chose tensorboardX which stands for tensorboard for X.


Installation
------------

To install the latest release of this package, simply run:

.. code-block:: sh

   pip install tensorboardX

For the latest development revision:

.. code-block:: sh

   pip install git+https://github.com/lanpa/tensorboard-pytorch.git

The tensorboard web server required to visualize the logs is provided as part of the
tensorflow software which can be obtained by following  the `instructions on the project
page <https://www.tensorflow.org/install>`_. Normally, the following command should
normally give you all required functionalities:

.. code-block:: sh

   pip install tensorflow


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
