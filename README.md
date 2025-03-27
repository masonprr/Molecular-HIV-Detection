# Predicting-HIV-Inhibition
AI Applications in Molecular Engineering: Predicting HIV Inhibition from Molecular Diagrams Using Schnet and Keras

**What We Did**
We implemented two models—one using the SchNetPack package, and another using a Keras model. Our main goal was to compare the predictive power of these two models, as shown in the notebook. We found that the SchNet model we used (with 64 n_in and 64 epochs) was able to predict 6,291 out of 6,487 samples effectively. In the report regarding the Keras models, this will be compared to their performance.

To implement the SchNet model, we used code provided by Dr. Ward along with the SchNetPack documentation. The main parameter we changed was converting the regression results into a classification output. SchNetPack is normally used for regression tasks, but our problem was one of classification—so we had to choose a cutoff for the regressor that would best suit our model. Since nearly all regression values came out between 0 and 1 (as expected, given that these were the only values in the HIV_active data column), we iterated over all values from 0 to 1 to the thousandth place and selected the optimal threshold.

For the Keras models, we first uploaded the data as saved PNG Matplotlib figures of the images used in the SchNet models. This changed the image sizes compared to those used in SchNet, and they were resized later. We built three models: the first with a Conv2D, pooling, flatten, and two dense layers, with each subsequent model adding one more dense layer. A dense layer with one unit and a sigmoid activation function was used as the final layer in each model, since we were performing binary classification. Each model was trained for 20 epochs with 20 steps per epoch. The RMSE of each model was also calculated, and their performance was compared in a graph at the end of the document.

**What We Learned**
From the SchNet analysis, we learned the challenges of data cleaning and how difficult it can be to get data into a usable format. Much of our time was spent debugging issues related to NA values in the dataset, which caused errors during model training. We also learned how to convert regression outputs into classification outputs by creating and optimizing a cutoff value for the regression data.

Additionally, we learned that training models can be time-consuming—sometimes taking hours per model. We gained experience using AI packages, which is likely to be useful in the future for projects in materials science or other fields. Finally, we became familiar with PyTorch, as SchNet is primarily implemented in PyTorch rather than TensorFlow.

From the Keras modeling, we learned about the challenges of image preprocessing and data formatting. We learned how to convert displayed images in Python into .png files, store them locally, categorize them, and import them into a Keras image data generator to format them for training.

Furthermore, we explored various methods of loading data into Keras, such as .flow_from_directory and .flow_from_dataframe. We also learned about different CNN layers and the importance of using a sigmoid activation function in the final layer for binary image classification.

**What We Would Do Next**
Future steps could include further optimizing our models by tuning more parameters. We were limited by the computing power of our local machines, and with access to a computing cluster, we could explore more ambitious configurations (more epochs, deeper networks, etc.). Training on a larger dataset from the start would also likely improve performance. This is another advantage of using a compute cluster—it would allow us to submit batch jobs and use more powerful machines to train on much larger datasets (we only trained on 1,000 samples, which already took a considerable amount of time). With more resources, we could test many more parameters and obtain a more optimal model.

As noted in the SchNet Jupyter notebook, we attempted a few ideas that didn’t work out. These were discussed in more detail during the recording. Essentially, after multiple attempts to add a sigmoid layer, we were unsuccessful. The different methods we tried and the errors encountered are documented in the notebook.

Next steps for the Keras models include using a larger subset of the HIV dataset and training the models over more epochs. In the end, the models were weak due to being trained on too small and too imbalanced a dataset. Over 90% of the training images were categorized as non-inhibitors, limiting the model’s ability to learn patterns in inhibitor molecules.

In addition to expanding the dataset, it would be beneficial to learn how to upload images into Keras using direct NumPy arrays generated from SMILES strings. This would speed up the data upload process and eliminate the need to download images onto personal devices.

Finally, we can increase model complexity by adding more layers. Some of the Keras models we encountered in our research were significantly more complex and produced better results—even on smaller or less sophisticated datasets. Experimenting with additional or different types of layers could be a powerful next step.
