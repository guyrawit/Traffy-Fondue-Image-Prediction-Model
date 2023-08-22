# Introduction

Traffy Fondue is an application that people can use to report problems
in Bangkok. Reporters can report issues by just taking a picture of the
issue and uploading it to the application system, which was designed to
be easy to use, and then the system will send it to the officer and
responsible team. Moreover, reporters can track their report and its
process on their mobile phones.

<img src="./attachments/$myfilename/media/image1.png"
style="width:6.01174in;height:2.04212in" />

So, my work is classifying each problem of each picture into 10 classes
of cases reported by Traffy Fondue.

-   Sanitary

-   Sewer

-   Stray

-   Canal

-   Light

-   Flooding

-   Electric

-   Traffic

-   Road

-   Sidewalk

With a data period until September 30th, 2022.

Noted : Original data from Kaggle with 33,143 Training images without
cleaning and 6,425 cleaned Testing images

<img src="./attachments/$myfilename/media/image2.png"
style="width:4.24596in;height:2.86215in"
alt="Graphical user interface, application Description automatically generated" />

# Data preparation

## Manual cleaned process

It took around 3 days to clean the training data because the data was so
messy with wrong labels, irrelevant images, and duplicated images.

Then, after I manually cleaned training images, I got the number of each
category as shown below.

<img src="./attachments/$myfilename/media/image3.png"
style="width:5.68995in;height:3.07177in"
alt="Chart, bar chart Description automatically generated" />

As you can see, "stray" has only around 500 images, but we cannot do
much about this because if we decrease the number of other categories’
images to 500, it’s not worth it. Therefore, I just focus on accuracy
whether that is okay or not.

## Example of noisy data 

<img src="./attachments/$myfilename/media/image4.png" style="width:2in;height:2in" alt="A person walking on a path next to a body of water Description automatically generated with low confidence" /> <p>Figure 1 complicated label image<p> 
<img src="./attachments/$myfilename/media/image5.jpg"
style="width:2in;height:2in"
alt="A picture containing outdoor Description automatically generated" /> <p>Figure 2 Multiple issues image</p>
<img src="./attachments/$myfilename/media/image6.png"
style="width:2in;height:2in"
alt="Text, letter Description automatically generated" /> <p>Figure 3 irrelevant image</p> <br>





## Train & Validation split

I’ve splited data with 80% for the training set and 20% for the
validation set with a fixed random seed (seed = 2022) as shown below.

<img src="./attachments/$myfilename/media/image7.png"
style="width:4.84399in;height:4.13563in"
alt="Graphical user interface, text, application Description automatically generated" />

## Normalization

Normalizing data is a must when we deal with image data because it
reduces the possibility of exploding gradients and improves convergence
speed. But a pre-trained model such as EfficientNet is included as part
of the model as shown below.

<img src="./attachments/$myfilename/media/image8.png"
style="width:4.16022in;height:1.89213in"
alt="Graphical user interface, text, application Description automatically generated" />*https://keras.io/examples/vision/image\_classification\_efficientnet\_fine\_tuning/*

## Data Augmentation

I put data augmentation as part of the model as shown below by using 4
layers of data augmentation.

<img src="./attachments/$myfilename/media/image9.png"
style="width:7.3in;height:0.95903in"
alt="Text Description automatically generated with medium confidence" />

<img src="./attachments/$myfilename/media/image10.png"
style="width:3.28616in;height:3.21676in" />

*Example of data Augmentation*

# Model

## Model baseline selection 

<img src="./attachments/$myfilename/media/image11.png"
style="width:4.65994in;height:3.89703in"
alt="Table Description automatically generated" /><img src="./attachments/$myfilename/media/image12.png"
style="width:4.65706in;height:3.84234in"
alt="Table Description automatically generated" />

I’ve started from a model that has parameters of not over 10 million and
has good accuracy, that is, EfficientNetB2 as the baseline model.

## Baseline model - EfficientnetB2

I’ve started from this one because It’s not too large and got 80.1%
accuracy on “ImageNet”. So, I used EfficientNetB2 as baseline model.

<img src="./attachments/$myfilename/media/image13.png"
style="width:4.74391in;height:1.7499in"
alt="Table Description automatically generated" />*https://keras.io/examples/vision/image\_classification\_efficientnet\_fine\_tuning/*

My model architecture begins with EfficientNetB2 with no trainable
parameters and is connected with a few layers consisting of a flattening
layer, a drop out layer, and a dense layer with 260 input sizes, as
recommended. So, I achieved a 0.76722 accuracy score on the test set as
shown below.

<img src="./attachments/$myfilename/media/image14.png">

## Other models – Efficientnet (B3, B4, V2B2 & V2B3)

I keep improve accuracy by trying larger EfficientNet based model and
tuning hyperparameters with limit resource and time consuming in the
same way I got better accuracy as shown below

<img src="./attachments/$myfilename/media/image15.png"
alt="Table Description automatically generated" />

*Note that models shown above are not in ranking order with accuracy
score.*

## Best model - EfficientnetV2-S

The best one is the pre-trained model efficientnetV2-S with transfer
learning connected with a few layers and fine-tuning parameters that you
can see in the image down below.

-   Input shape (380, 380, 3) with 24 batch size

    -   Larger input size performs better score but take more computing
        resource and time consuming.

-   Pretrain model EfficientnetV2-S with 30 trainable layers from the
    top layers exclude batch normalization layers

    -   I tried to train some layers of EfficientNetV2-S pretrained
        weight and got better result than freeze them all.

> <img src="./attachments/$myfilename/media/image16.png"
> style="width:4.9108in;height:3.70085in"
> alt="Text Description automatically generated" />

*List of EfficientNetV2-S layers were adjusted to trainable*

-   Average pooling layer on top the Efficientnet layer

-   Batch Normalization and then dense to embedding vector

-   2 40 percent dropout and 1024 256 dense layers

-   And final layer is 10 category of images dense layer

> <img src="./attachments/$myfilename/media/image17.png"
> style="width:4.17248in;height:3.86248in"
> alt="Table Description automatically generated" />

After I trained over 11 epochs, I reached 0.80142 accuracy on the
validation set (epoch 8), and I made an early stop when accuracy did not
improve in 3 epochs.

<img src="./attachments/$myfilename/media/image18.png"
style="width:7.3in;height:1.52639in"
alt="Text Description automatically generated with medium confidence" />

And reached 0.80107 on test set

<img src="./attachments/$myfilename/media/image19.png"
style="width:7.3in;height:0.70486in" />

### Training and Validation loss 

### <img src="./attachments/$myfilename/media/image20.png"
style="width:7.26757in;height:3.7285in"
alt="Chart, line chart Description automatically generated" /> 

### Confusion Metric & F1- score

<img src="./attachments/$myfilename/media/image21.png"
style="width:3.09938in;height:2.86508in"
alt="Chart, scatter chart Description automatically generated" />
<img src="./attachments/$myfilename/media/image22.png"
style="width:4.00288in;height:2.34359in"
alt="Table Description automatically generated" />

# Results

## My submissions score

After I reached 0.80107 with the EfficientNetV2-S (380, 380, 3) model, I
tried a method called ensemble that uses many models to predict together
and uses the most votes on each picture as the predicted answer, but if
all models predict differently, I just used the model that got the
maximum accuracy. So I used 6 models with 6 results from my models that
I’ve trained, as you can see below. As you see, accuracy improves much
more with the ensemble method.

<img src="./attachments/$myfilename/media/image15.png"
style="width:7.3in;height:4.8625in"
alt="Table Description automatically generated" />

## Kaggle Leaderboard 

<img src="./attachments/$myfilename/media/image24.png"
style="width:7.3in;height:2.21597in"
alt="Graphical user interface, application Description automatically generated" />

# Discussion

## Error analysis

Although I reached Top-1 Kaggle score, I think it could be improved by
this list here.

-   Use larger model and more complexity

    -   Due to my experiment, when I used a larger model (higher
        parameters), accuracy improved a lot, but I got limited
        computing resources, so I could not train a larger model than
        EfficientNetV2-S, which contains 21 million parameters (25 GB of
        RAM and NVIDIA tesla t4 GPU).

-   Clean more data

    -   I think that cleaning data is the most important thing in this
        case because data is so noisy. For example, if data is
        mislabeled, it can lead to model misunderstanding. So, we can
        start from the heat map confusion metric to find which
        categories have the most missed predictions.

# Conclusion

This work is a classification of 10 classes of images of the Traffy
Fondue application, so there are many things that can affect the model's
performance and accuracy score. First and foremost, data preparation is
the most important aspect of this job because if you have mislabeled
data in your training set, your model may misinterpret which image is
correct or incorrect and thus perform poorly on the test set. The next
one is choosing the largest model as you can because after I tried many
models, the accuracy score improved when the model is larger (more
parameters). I used EfficientNetV2-S, which contains around 21 million
parameters . And the last one is fine tuning model parameters such as
batch size, input size, top up layers, adjustable pre-trained weight and
others. After I tried to tune hyperparameters, accuracy and F1 score
improved, but I think this process is harder because it is
time-consuming and just like trial and error. Finally, the ensemble
method is the last thing I’ve tried, as you've seen before. It’s
improved my results just like 10 people better than one.
