# Text Classification

For this lab, we'll use script mode with Keras / TensorFlow for Object Detection.  This is a very common problem that many customers encounter.  Given some example text documents, how can you classify them into catagories?  We'll use an open dataset with word embeddings to train a custom classification task.  Let's get started!

Follow these steps:

### 1. Upload files for the lab

1\. [Click this link](resources/keras_glove.zip) to download a compressed folder to your local machine.

2\. With your notebook instance open, Upload in the upper right of the screen.  

3\. Navigate to the downloaded zip file, click Open. 

4\. Click the blue upload button to upload

5\. With the zip file uploaded, Open a new terminal on the notebook instance.  Click "New" in the upper right of the screen and select "Terminal" at the end of the list.

6\. With the new terminal open, change directories and unzip the file
```
cd SageMaker
mkdir text_classification
unzip keras_glove.zip -d ./text_classification
```

7\. You are now ready to begin the notebook:  click the notebook's file name to open it.

8. If you are familiar with Jupyter notebooks, you can skip this step.  Otherwise, please expand the instructions below.

??? optional-class "Jupyter notebook instructions (expand for details)"
    - Jupyter notebooks tell a story by combining explanatory text and code. There are two types of "cells" in a notebook:  code cells, and "markdown" cells with explanatory text.  
    - You will be running the code cells.  These are distinguished by having "In" next to them in the left margin next to the cell, and a greyish background.  Markdown cells lack "In" and have a white background.
    - To run a code cell, simply click in it, then either click the **Run Cell** button in the notebook's toolbar, or use Control+Enter from your computer's keyboard.  
    - It may take a few seconds to a few minutes for a code cell to run (an asterick will appear in "In[ ]" when running and will change to a number once execution has completed).  Please run each code cell in order, and only once, to avoid repeated operations.  For example, running the same training job cell twice might create two training jobs, possibly exceeding your service limits.

??? optional-class "Approximate times (expand for details)"
    - Training the text classification model takes about 10 minutes
    - The total lab takes about 60 minutes

!!! Done
    Fantastic! You’ve just trained a custom text classification model!  Woot!


