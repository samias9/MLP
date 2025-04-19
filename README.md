### learning rate
taux d’apprentissage (learning rate)


Le processus d’apprentissage est sensible au taux d’apprentissage.
Au début, les sorties du modèle étaient médiocres, ce qui m’a fait soupçonner un bug.
Cependant, il suffisait de modifier la valeur du taux d’apprentissage pour résoudre le problème.

### num timesteps

nombre d'étapes
Un processus de diffusion plus long donne de meilleurs résultats.
Avec moins d’étapes, le dinosaure est incomplet, il manque des points en haut et en bas.

### variance schedule

Le planning quadratique ne donne pas de meilleurs résultats.

## References

* The dino dataset [Datasaurus Dozen](https://www.autodesk.com/research/publications/same-stats-different-graphs) data.
* HuggingFace's [diffusers](https://github.com/huggingface/diffusers) library.
* lucidrains' [DDPM implementation in PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
* Jonathan Ho's [implementation of DDPM](https://github.com/hojonathanho/diffusion).
* InFoCusp's [DDPM implementation in tf](https://github.com/InFoCusp/diffusion_models).
