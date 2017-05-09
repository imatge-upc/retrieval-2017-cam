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
Recently lots of works have proven that using Convolutional Neural Networks as feature extractors is very effective at tackling image retrieval tasks. In our work we explore encoding images based on their predicted semantics, building descriptors that gather more relevant knowledge about the scenes. We propose a retrieval pipeline where we employ Class Activation Maps to spatially weight convolutional features given the objects location. This class activation maps can be further exploited in a post re-ranking stage where they can provide an easy manner to compute regions of interest. Our experiments on two publicly available datasets, Oxford5k and Paris6k, demonstrate that our system is competitive and even outperforms the current state-of-the-art in off-the-shelf image retrieval.

![Vgg_cams](https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/figs/vgg_cam.png?raw=true)


![Encoding_pipeline](https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/figs/Pipeline.png?raw=true)

## Slides

[Slides](https://github.com/imatge-upc/retrieval-2017-icmr/blob/master/slides/Class-Weighted%20Convolutional%20Features%20for%20Image%20Retrieval%20.pdf)

## Publication

## Results 

![Comparison with State of the Art](https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/figs/StateArt1.png)



![Comparison with State of the Art - QE & RE](https://github.com/imatge-upc/Class-Weighted-Convolutional-Features-for-Image-Retrieval/blob/master/figs/StateArt2.png)

## Code Usage

### Prerequisites
We have used Keras running over Theano to perform the experiments. 

Was done previous to Keras 2.0 but should work with that version as well. 

Python packages necessary specified in *requirements.txt* run:

```
 pip install -r requirements.txt
 
```

Our Experiments have been carried out in these datasets:

* [Oxford Buildings](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) (and Oxford 105k).

* [Paris Buildings](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) (and Paris 106k).

Here we provide the weigths of the model:

* [CAM-VGG-16 Weights](https://drive.google.com/open?id=0BwotWbbE50RQLWtPZ1VWTDYzUlE)

### How to run the code?

First thing to do (important!) is setting the path of your images and model weights. We provide lists (also modify path! - Find and Replace) that divide images in vertical and horizontal for faster processing. At the beggining of each script there are some parameters that can be tuned like image preprocessing. I have added a parser for arguments, at the beginning of each script it is shown an example of how to run them. 

#### Feature Extraction

Both scripts extract Class-Weighted Vectors. The first one is used for the original datasets. The second for the distractors.  You tune the preprocessing parameters of the images as well as the number of Class-Weighted Vectors extracted. In "Online Aggregation" the order of the stored vectors is the imagenet class order, while in "Offline Aggregation" the order of the vector is from class more probable to less probable (predicted by the network). 

* A_Oxf_Par_Feat_CAMs_Extraction.py 
* A_Dist_Feat_CAMs_Extraction.py


#### Aggregation, Ranking and Evaluation

In both scripts you can choose the dataset you want to evaluate and if use query expansion or re-ranking. The first one is for offline aggregation. The second one performs aggregation at the moment of testing.

* B_Offline_Eval.py
* B_Online_Aggregation_Eval.py

```

B_Online_Aggregation_Eval.py --d <dataset> --nc_q <nclasses_query> --pca <n_classes_pca> --qe <n_query_exp> --re <n_re_ranking> --nc_re <n_classes_re_ranking>

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
