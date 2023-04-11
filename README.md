# PersonClassification
Project Work for Internship

It consists of two python files:
modelCreation.py

This file is responsible for using transfer learning to create a resnet18 instance model trained on the images.

predict.py

Through this file we can test particular image by providing it as input and test what the model has predicted.
I also saved the weights of the model on my machine so that I can use it to prediction and testing separately.

NOTE: Since the dataset is too small consisting of only 16 images, 4 belonging to each class, there is very little room for any productive technique implementaion.
Possibly data augmentation can be employed but that won't yield to any significant improvement either. More training would lead to overfitting on the training set
which is very-very limited. And certainly there is no scope for hyperparameter tuning either. Ensemble methods such as averaging the predictions of multiple models
can help improve the accuracy and robustness of the classification results but not to any significant extent under such a small dataset.

However, if asked I can move with the alternative approches I have suggested.

I have also added accuracy results before/after adding k_fold_validation. And another image where the model correctly predicts Mr. AR Rahman. (Label:0)
