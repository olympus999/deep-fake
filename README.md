Kaggle competition this code is made for: https://www.kaggle.com/c/deepfake-detection-challenge/overview

Notebook named: **deepfake-detection-challenge.ipynb**  
Here is code for training a model to separate manipulated and non-manipulated frames.  
I never intended to put it up so it has a bunch of unnecessary code and a lot of logging.  
Also, viewing the notebook in github us worse than when using jupyter-notebook with plugins.  
In this dataset we have same the frame with manipulated face and non-manipulated face.  
So in the same batch we can feed model an image which has been manipulated  
and an image which has not been manipulated. This way model should learn  
the features of the manipulated images and successfully classify those images. 

As all the frame manipulations are done on the faces, we benefit by cropping the faces  
and working with those images. For the training I went with Pytorch (previously I have used Keras only)  
as there already was a good face detection already implemented in Pytorch.   
The library used for face detection: https://github.com/timesler/facenet-pytorch  

For maximum speed I tried to implement python 3.8 multiprocessing.shared_memory.  
Using the shared_memory we can pass "pointers" to functions instead of massive image arrays.  
This should lead to considerable increase in training speed for this and future projects as  
in many cases CPU is the bottle neck (due to augmentation, cropping etc).  
But in this case I was using Windows machine and ran into OS error:  
"The paging file is too small for this operation to complete".  
I could not figure it out before the competition ended. But this is a good place  
to continue debugging the issue in the future. + I think it might work in Linux.
