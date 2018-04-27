Sound Classification
===================

This recurrent neural network tries to differentiate between sounds. The RNN is trained by using example data provided in the repo. in Sound-Data folder.

----------

#### <i class="icon-down-big"></i> Installation

	> - Clone Repository
	> - Install Dependencies

#### <i class="icon-ccw"></i> Training

	> - python main.py -m train

#### <i class="icon-right-big"></i> Testing

	> - python main.py -m pred -u <model ID from trained model here>

Dependiencies
-------------------

> tensorflow <br>
> numpy <br>
> librosa <br>
> matplotlib <br>
> tqdm <br>
> uuid <br>

Example
-------------------
The RNN successfully classificated 1.wav as label 0 in class dictionary
![img_pred](https://image.ibb.co/hN5OUx/Unbenannt.png)
