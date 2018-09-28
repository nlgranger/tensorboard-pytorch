Tutorial
========

.. currentmodule:: tensorboardX


TensorboardX follows the same mechanisms as `Tensorboard
<https://www.tensorflow.org/guide/summaries_and_tensorboard>`_: instrumentation code is
added to your scripts in order to generates log data at runtime, and the Tensorboard web
server is used to reload and display it.


Creating a summary writer
-------------------------

Before logging anything, we need to create an instance of :class:`SummaryWriter` in the
training script. A writer is tied to a log directory and all data measurements go through
this writer. Creating a writer can be done with:

.. code-block:: python

    from tensorboardX import SummaryWriter

    # Create writer object, logs will be saved in 'runs/exp-1'
    writer = SummaryWriter('runs/exp-1')

    # Alternatively, use an auto generated name, the directory will be
    # something like 'runs/Aug20-17-20-33'
    writer = SummaryWriter()

    # or use an auto generated name with the comment as a suffix:
    # for example 'runs/Aug20-17-20-33-3x learning rate'
    writer = SummaryWriter(comment='3x learning rate')

.. admonition:: Merging plots

   Tensorboard can monitor multiple experiments simultaneously by fusing the logs from
   several folders. Each time you re-run the experiment with different settings, you
   should change the name of the log directory such as `runs/exp1`, `runs/exp2` so that
   you can easily compare different experiment settings by running tensorboard on the
   `runs` directory. You can also nest the subfolders by having one writer for
   `runs/exp1/training` and one `runs/exp1/validation` for example.


General api format
------------------

Following the `tensorboard API
<https://www.tensorflow.org/versions/r1.11/api_guides/python/summary>`_, most of the
writer methods follow the pattern: :code:`writer.add_???(tag_name, value, global_step)`

where `tag_name` is a common name shared by all measurements of a given quantity,
`value` contains the data for one particular measurement, and `global_step` provides
its iteration number.

Contrary to Tensorflow, this will not define an operation but instantly export the value.

The command will fail when given PyTorch *cuda* Tensor in GPU memory.
Remember to extract the value to local memory before by running ``object.detach().cpu()``
when necessary.

.. admonition:: Grouping plots

   Usually, there are many quantities to track during an experiment. For example, when
   training GANs you would log the losses of the generator and the discriminator. If the
   loss is composed of two other loss terms, say L1 and MSE, you might want to log the
   values for each as well. In this case, you can write the tags as `'Gen/L1'`,
   `'Gen/MSE'`, `'Desc/L1'`, `'Desc/MSE'`. Tensorboard will group together these plots
   into two sections `Gen` and `Desc`.

   Note that grouping differs from *merging* plot (see above).


Log scalars
-----------

To log a scalar value, use :meth:`SummaryWriter.add_scalar`:

.. code-block:: python

    writer.add_scalar('tagname', value, global_step)

A scalar value is the most simple data type to deal with. Mostly we save the loss value of
each training step, or the accuracy after each epoch. Sometimes I save the corresponding
learning rate as well. Since logging scalar values is a cheap operation, you can log
anything deemed important without fearing noticeable slowdowns.


Log images
----------

Use :meth:`SummaryWriter.add_image` to log an image.

.. code-block:: python

    writer.add_image('tagname', im, global_step)

Images are represented by 3-dimensional tensors with size `[3, height, width]`, where the
first dimension corresponds to the red, green, blue channels of an image. If you need to
display multiple images, you may use :func:`torch:torchvision.utils.make_grid` to arrange
them onto a grid and then send the result to :meth:`SummaryWriter.add_image`.

.. Note::

   Remember to normalize your image values in range [0, 1].


Log histograms
--------------

To save a histogram, you actually need a numpy array containing the values to aggregate
to :meth:`SummaryWriter.add_histogram`:

.. code-block:: python

    writer.add_histogram('name', values, global_step)

Saving histogram is a more expensive operation, both in terms of computation time and
storage. You might want to limit the use of this function if you experience slowdowns.


Log computation graphs
----------------------

Graph drawing helps to visualize the different mathematical operations that compose a
model.

To generate the graph, :func:`SummaryWriter.add_graph` traverses the model recursively starting
from its inputs and drawing each encountered nodes.

This function uses the model (an instance of :class:`torch:torch.nn.Module`) and dummy
inputs:

.. code-block:: python

    model = MyNet()
    inputs = torch.rand(*input1_shape), torch.rand(*input2_shape)

    writer.add_graph(model, inputs)

By default, pytorch omits input vectors from the back-propagation graph. To keep input
nodes visible on the graph, pass an additional parameter ``requires_grad=True`` when
creating the input tensor. See
https://github.com/lanpa/tensorboard-pytorch/blob/master/demo_graph.py for complete
example.


Log audio
---------

Use :meth:`SummaryWriter.add_audio`. Currently the sampling rate of the this function is
fixed at 44100 KHz, single channel. The input of the add_audio function is a one
dimensional array, with each element representing the consecutive amplitude samples. For a
2 seconds audio, the input x should have 88200 elements. Each element should lie in [-1,
1].


Log embeddings
--------------

*What is an embedding?* embeddings usually refer to a mapping of input observations into
real vector data points. To convert high dimensional data into human perceptible 3D data.
Tensorboard provides PCA and t-sne dimensionality reduction techniques which help
visualize data in 2D or 3D while preserving most of the information about the
distribution of embedded points, for example the proximity between points.

What you need to do is provide is an array of row vector data point to
:func:`SummaryWriter.add_embedding`, and Tensorboard will do the rest for you:

.. code-block:: python

    model = writer.add_embedding(model, embeddings)

To make the visualization more informative, you can pass optional metadata or images
corresponding to each data points. In this way you can see that neighboring point have
similar label and distant points have very different label (semantically or visually).
See https://github.com/lanpa/tensorboard-pytorch/blob/master/demo_embedding.py for
detailed example.


Performance considerations
--------------------------

Logging is cheap, but displaying is expensive. From my experience, if there are 3 or more
experiments to show at a time and each experiment have, say, 50K points, tensorboard might
need a lot of time to render the data.
