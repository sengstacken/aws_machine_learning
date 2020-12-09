# Object Detection with Script Mode

For this lab, we'll explore Amazon SageMaker - Script Mode.  This time we'll utilize MXNet to train a custom object detection model.  To train this model we'll use the hyperparameter optimization job.  This will automatically adjust the hyperparameters to select the best values.    

Follow these steps:

### 1. Upload files for the lab

1\. [Click this link](resources/yolodice.zip) to download a compressed folder to your local machine.

2\. With your notebook instance open, Upload in the upper right of the screen.  

3\. Naviagate to the downloaded zip file, click Open. 

4\. Click the blue upload button to upload

5\. With the zip file uploaded, Open a new terminal on the notebook instance.  Click "New" in the upper right of the screen and select "Terminal" at the end of the list.

6\. With the new terminal open, change directories and unzip the file
```
cd SageMaker
mkdir yolo
unzip yolodice.zip -d ./yolo
```

7\. You are now ready to begin the notebook:  click the notebook's file name to open it.


!!! Info "Note!"
    Training the model for this example typically takes about 45 minutes minutes 


!!! Done
    Awesome! You’ve just trained a custom convolutional neural network!  

