# Learning Low-Dose CT image denoising from an abstract dataset

Contact: [Raphaël Achddou](mailto:raphael.achddou@epfl.ch), [Paulo Ribeiro](mailto:paulo.ribeirodecarvalho@epfl.ch)

## Introduction

Computed Tomography (CT) images are particularly helpful for medical diagnosis of multiple diseases, especially for
brain lesions analysis. While High-Dose CT images are easy to interpret thanks to their sharp contrast, the chemical
product and the X-rays radiation used to enhance the contrast is invasive and toxic for the patient. In that regard,
most CT imaging techniques are performed with low-dose. This implies that the resulting image is less contrasted and
that the signal to noise ratio is much higher[1].

In that regard, CT image denoising is a crucial step to visualize and analyze such images. Most classic image processing
algorithms have been transferred to the medical image domain, producing satisfactory results [1]. The results have been
boosted by a large margin by the introduction of neural nets for image restoration. [2,3]

However, one key limitation for the interpretability of the produced method is that neural networks hallucinate patterns
and textures they have seen in the training set. Therefore, these methods are not really trustworthy for radiologists.

In order to leverage the expressive power and denoising capacity of deep neural networks without recreating patterns
which have been seen during training, the idea is to train on synthetic abstract images which do not directly contain
the patterns observed in real CT images. That way, our network won’t be biased to reproduce expected patterns, while
maintaining good performances.

## Tasks

In this semester project , we will try to create a database of abstract dead leaves images mimicking the statistics of
real images [4,5] study the noise distribution of real CT images by using a real dataset of noisy images train a
denoising network with the simulated ground truth and noisy images quantify the hallucinations made by the network
trained on real images vs simulated images [6] establish a test protocol for our network

## Deliverables

Deliverables should include code, well cleaned up and easily reproducible, as well as a written report, explaining the
models, the steps taken for the project and the performances of the models.

## Tags 

Python and PyTorch, basics of image processing

## References

1) A review on CT image noise and its denoising, 2018, Manoj Diwakara, Manoj Kumarb
2) Low-Dose CT Image Denoising Using a Generative Adversarial Network With a Hybrid Loss Function for Noise Learning, 2020, YINJIN MA et al
3) Investigation of Low-Dose CT Image Denoising Using Unpaired Deep Learning Methods, 2021, Zeheng Li, Shiwei Zhou, Junzhou Huang, Lifeng Yu, and Mingwu Jin
4) Occlusion Models for Natural Images: A Statistical Study of a Scale-Invariant Dead Leaves Model, 2001, Ann B. Lee, David Mumford, Jinggang Huang
5) Synthetic images as a regularity prior for image restoration neural networks, 2021, Raphael Achddou, Yann Gousseau, Said Ladjal
6) Image Denoising with Control over Deep Network Hallucination, 2022, Qiyuan Liang, Florian Cassayre, Haley Owsianko, Majed El Helou, Sabine Süsstrunk