# Fast Gradient Sign Method (FGSM)

This is a rough implementation of the "fast gradient sign method" from the [Explaining and Harnessing Adversarial Examples paper by Ian Goodfellow et al.](https://arxiv.org/abs/1412.6572) on the MNIST dataset.

## Results

As you can see, even though both images are remarkably similar the model is incorrect after introducing a small perturbation using the fast gradient sign method.

| Image                                             | Prediction | Actual | Correct? |
| ------------------------------------------------- | ---------- | ------ | -------- |
| <img src="fgsm_files/fgsm_0_orig.png" width=60 /> | 6          | 6      | True     |
| <img src="fgsm_files/fgsm_0_adv.png"  width=60 /> | 2          | 6      | False    |
| <img src="fgsm_files/fgsm_1_orig.png" width=60 /> | 8          | 8      | True     |
| <img src="fgsm_files/fgsm_1_adv.png"  width=60 /> | 2          | 8      | False    |
| <img src="fgsm_files/fgsm_2_orig.png" width=60 /> | 5          | 5      | True     |
| <img src="fgsm_files/fgsm_2_adv.png"  width=60 /> | 3          | 5      | False    |
| <img src="fgsm_files/fgsm_3_orig.png" width=60 /> | 0          | 0      | True     |
| <img src="fgsm_files/fgsm_3_adv.png"  width=60 /> | 2          | 0      | False    |
| <img src="fgsm_files/fgsm_4_orig.png" width=60 /> | 3          | 3      | True     |
| <img src="fgsm_files/fgsm_4_adv.png"  width=60 /> | 1          | 3      | False    |
