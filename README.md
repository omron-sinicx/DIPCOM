# DIPCOM: Learning Diffusion Policies from Demonstrations For Compliant Contact-rich Manipulation

Video: https://youtu.be/77tUVfyqHrw

Project page: https://omron-sinicx.github.io/DIPCOM/

## Installation

1. [Optional] clone robosuite dependencies for simulation
  ```shell
  git clone https://github.com/omron-sinicx/robosuite.git -b dipcom
  ```

2. use `pixi` for easy setup

- Install `pixi`
  ```shell
  curl -fsSL https://pixi.sh/install.sh | bash
  ```

- Setup pixi environment
  ```shell
  pixi install

  pixi shell
  ```

Then check the [dipcom/README.md](dipcom/README.md) for more detailed instructions.

## Docker
Alternatively you can use our docker setup:
1. Build the docker image
```shell
./BUILD-DOCKER-IMAGE.sh
```
2. Run container
```shell
./RUN-DOCKER-CONTAINER.sh
```
3. Follow the steps in the "**Installation**" section above.



### Authors:
- Malek Aburub
- Cristian C. Beltran-Hernandez
- Tatsuya Kamijo
- Masashi Hamaya


### Acknowledgements:
This project builds upon the open-source [LeRobot - HuggingFace](https://github.com/huggingface/lerobot)

If you find this project useful, consider citing it.
```
  @article{aburub2024learning,
      title={Learning Diffusion Policies from Demonstrations For Compliant Contact-rich Manipulation}, 
      author={Malek Aburub and Cristian C. Beltran-Hernandez and Tatsuya Kamijo and Masashi Hamaya},
      year={2024},
      eprint={2410.19235},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.19235}, 
}
```
