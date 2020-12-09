# Object Detection Notebook

For this lab, we'll work with an image classification and detection example notebook. In particular, we'll use the Amazon-provided object detection algorithm, which is a supervised learning algorithm that takes an image as input and classifies it into one of multiple output categories as well as the bounding boxes for the detected classes. It uses a convolutional neural network (ResNet) that can be trained from scratch, or trained using transfer learning when a large number of training images are not available. Even if you don't have experience with neural networks or object detection, SageMaker's object detection algorithm makes the technology easy to use, with no need to design and set up your own neural network.  

Follow these steps:

1. In the sagemaker notebook interface, navigate to the **SageMaker Examples** tab.  
2. Find the line item for **object_detection_image_json_format.ipynb** under the "Introduction to Amazon Algorithms"
3. Click the orange "Use" button and "Create copy" button on the pop up window.
4. You are now ready to begin the notebook: 
5. If you are familiar with Jupyter notebooks, you can skip this step.  Otherwise, please expand the instructions below.

??? optional-class "Jupyter notebook instructions (expand for details)"
	- Jupyter notebooks tell a story by combining explanatory text and code. There are two types of "cells" in a notebook:  code cells, and "markdown" cells with explanatory text.  
	- You will be running the code cells.  These are distinguished by having "In" next to them in the left margin next to the cell, and a greyish background.  Markdown cells lack "In" and have a white background.
	- To run a code cell, simply click in it, then either click the **Run Cell** button in the notebook's toolbar, or use Control+Enter from your computer's keyboard.  
	- It may take a few seconds to a few minutes for a code cell to run (an asterick will appear in "In[ ]" when running and will change to a number once execution has completed).  Please run each code cell in order, and only once, to avoid repeated operations.  For example, running the same training job cell twice might create two training jobs, possibly exceeding your service limits.

??? optional-class "Approximate times (expand for details)"
	- Training the object detection model takes about 10 minutes
	- Keep in mind that this is relatively short because transfer learning is used rather than training from scratch, which could take many hours!
	- Creating the image classification model endpoint takes about 10 minutes
	- The total lab takes about 60 minutes


!!! Done
    Fantastic! You’ve just trained a convolutional neural network!!
	(Your friends will be impressed)

