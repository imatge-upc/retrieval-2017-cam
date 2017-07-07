# Class-Weighted Convolutional Features for Image Retrieval

| ![ALbert Jimenez][AlbertJimenez-photo]  | ![Xavier Giro-i-Nieto][XavierGiro-photo]  |  ![Jose M. Alvarez][JoseAlvarez-photo] | 
|:-:|:-:|:-:|
| [Albert Jimenez][AlbertJimenez-web]  | [Xavier Giro-i-Nieto][XavierGiro-web]   |[Jose M.Alvarez][JoseAlvarez-web] | 

[AlbertJimenez-web]: https://www.linkedin.com/in/albertjimenezsanfiz/
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro
[JoseAlvarez-web]: http://www.josemalvarez.net

[AlbertJimenez-photo]: https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/authors/AlbertJimenez.png?raw=true "Albert Jimenez"
[XavierGiro-photo]: https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/authors/XavierGiro.jpg?raw=true "Xavier Giro-i-Nieto"
[JoseAlvarez-photo]: https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/authors/JoseAlvarez.png?raw=true "Jose Alvarez"

A joint collaboration between:

| ![logo-gpi] | ![logo-data61] |
|:-:|:-:|
|[UPC Image Processing Group][gpi-web] | [Data61][data61-web]|
 
[gpi-web]: https://imatge.upc.edu/web/ 
[data61-web]: http://www.data61.csiro.au

[logo-data61]: https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/logos/data61.png?raw=true "Data 61"
[logo-gpi]: https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/logos/gpi.png?raw=true "UPC Image Processing Group"


## Abstract 
Image retrieval in realistic scenarios targets large dynamic datasets of unlabeled images. In these cases, training or fine-tuning a model every time new images are added to the database is neither efficient nor scalable. Convolutional neural networks trained for image classification over large datasets have been proven effective feature extractors for image retrieval. 

The most successful approaches are based on encoding the activations of convolutional layers, as they convey the image spatial information. In this paper, we go beyond this spatial information and propose a local-aware encoding of convolutional features based on semantic information predicted in the target image. To this end, we obtain the most discriminative regions of an image using Class Activation Maps (CAMs). CAMs are based on the knowledge contained in the network and therefore, our approach, has the additional advantage of not requiring external information. In addition, we use CAMs to generate object proposals during an unsupervised re-ranking stage after a first fast search. 

Our experiments on two public available datasets for instance retrieval, Oxford5k and Paris6k, demonstrate the competitiveness of our approach outperforming the current state-of-the-art when using off-the-shelf models trained on ImageNet.


![Encoding_pipeline](https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/figs/pipeline.png?raw=true)

## Slides

<center>
<iframe src="//www.slideshare.net/slideshow/embed_code/key/3dG0uuBHScqPTa" width="595" height="485" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:5px; max-width: 100%;" allowfullscreen> </iframe> <div style="margin-bottom:5px"> <strong> <a href="//www.slideshare.net/xavigiro/class-weighted-convolutional-features-for-image-retrieval" title="Class Weighted Convolutional Features for Image Retrieval " target="_blank">Class Weighted Convolutional Features for Image Retrieval </a> </strong> from <strong><a target="_blank" href="https://www.slideshare.net/xavigiro">Xavier Giro</a></strong> </div>
</center>

## Publication

(under review)

## Results 

### Comparison with State of the Art
![Comparison with State of the Art](https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/figs/StateArt1.png?raw=true)

![Comparison with State of the Art - QE & RE](https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/figs/StateArt2.png?raw=true)

### Qualitative Results 
![Qualitative Results of the Search](https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/figs/Qualitative.png?raw=true)

## Code Usage

In this repository we provide the code used in our experiments. 
VGG-16 CAM experiments where carried out using [Keras](keras.io) running over [Theano](http://deeplearning.net/software/theano/).
DenseNet and ResNet experiments were carried out using [PyTorch](http://pytorch.org). 

In the next Section we explain how to run the code in Keras+Theano. To run the experiments using PyTorch, the requirements are the same plus having installed Pytorch and the torchvision package.

### Prerequisites
Was done previous to Keras 2.0 but should work with that version as well. 

Python packages necessary specified in *requirements.txt* run:

```
 pip install -r requirements.txt
 
```

Our Experiments have been carried out in these datasets:

* [Oxford Buildings](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) (and Oxford 105k).

* [Paris Buildings](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) (and Paris 106k).

Here we provide the weigths of the model (paste them in models folder):

* [CAM-VGG-16 Weights](https://drive.google.com/open?id=0BwotWbbE50RQMlFnZ1N3c2tvUm8)

### How to run the code?

First thing to do (important!) is setting the path of your images and model weights. We provide lists (also modify path! - Find and Replace) that divide images in vertical and horizontal for faster processing. At the beggining of each script there are some parameters that can be tuned like image preprocessing. I have added a parser for arguments, at the beginning of each script it is shown an example of how to run them. 

#### Feature Extraction

Both scripts extract Class-Weighted Vectors. The first one is used for the original datasets. The second for the distractors.  You tune the preprocessing parameters of the images as well as the number of Class-Weighted Vectors extracted. In "Online Aggregation" the order of the stored vectors is the imagenet class order, while in "Offline Aggregation" the order of the vector is from class more probable to less probable (predicted by the network). 

* A_Oxf_Par_Feat_CAMs_Extraction.py 
* A_Dist_Feat_CAMs_Extraction.py

```

A_Oxf_Par_Feat_CAMs_Extraction.py  -d <dataset> -a <agreggation>

```

#### Aggregation, Ranking and Evaluation

In both scripts you can choose the dataset you want to evaluate and if use query expansion or re-ranking. The first one is for offline aggregation. The second one performs aggregation at the moment of testing.

* B_Offline_Eval.py
* B_Online_Aggregation_Eval.py

```

B_Online_Aggregation_Eval.py -d <dataset> --nc_q <nclasses_query> --pca <n_classes_pca> --qe <n_query_exp> --re <n_re_ranking> --nc_re <n_classes_re_ranking>

```


## Aknowledgements
We would like to specially thank Albert Gil and Josep Pujal from our technical support team at the Image Processing Group at UPC.

| ![AlbertGil-photo]  | ![JosepPujal-photo]  |
|:-:|:-:|
| [Albert Gil](AlbertGil-web)  |  [Josep Pujal](JosepPujal-web) |

[AlbertGil-photo]: https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/authors/AlbertGil.jpg?raw=true "Albert Gil"
[JosepPujal-photo]:https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/authors/JosepPujal.jpg?raw=true "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/retrieval-2017-icmr/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:albertjimenezsanfiz@gmail.com>.
