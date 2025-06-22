# Fast Gradient Sign Method (FGSM)

This is a rough implementation of the "fast gradient sign method" from the [Explaining and Harnessing Adversarial Examples paper by Ian Goodfellow et al.](https://arxiv.org/abs/1412.6572) on the MNIST dataset.

## Results

As you can see, even though both images are remarkably similar the model is incorrect after introducing a small perturbation using the fast gradient sign method.

| Image                                           | Prediction | Actual | Correct? |
| ----------------------------------------------- | ---------- | ------ | -------- |
| <img src="fgsm_files/fgsm_15_0.png" width=60 /> | 4          | 4      | True     |
| <img src="fgsm_files/fgsm_15_2.png" width=60 /> | 2          | 4      | False    |
