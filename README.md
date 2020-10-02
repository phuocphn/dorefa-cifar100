# Pytorch CIFAR100 + DoReFa 

This is a Pytorch implementation of DoreFa with CIFAR-100 dataset.

## Installation

Use [git clone](https://pip.pypa.io/en/stable/) to run this project.

```bash
git clone https://github.com/phuocphn/dorefa-cifar100
```

## Usage

**Train full precision baseline model.**
```shell
mkdir logs
python main.py --bit 32 \
--log_name resnet18_w32a32-ceres-01 \
--arch resnet18 --wd 5e-4 \
--max_epochs 250 >> logs/resnet18_w32a32-ceres-01.txt
```
**Train W2A2 model.**
```shell
mkdir logs
 python main.py --bit 2 \
--log_name resnet18_w2a2-ceres-01 \
--arch resnet18 --wd 5e-4 \
--max_epochs 250 >> logs/resnet18_w2a2-ceres-01.txt
```
**Train W4A4 model.**
```shell
mkdir logs
python main.py --bit 4 \
--log_name resnet18_w4a4-ceres-01 \
--arch resnet18 --wd 5e-4 \
--max_epochs 250 >> logs/resnet18_w4a4-ceres-01.txt
```

**Train W2A2 model. (DoreFa for weight, PACT for activation) **
```shell
mkdir logs
python main.py --bit 2 \
--log_name resnet18_w2a2-pact--ceres-01 \
--arch resnet18 --wd 5e-4 \
--max_epochs 250 >> logs/resnet18_w2a2-pact-ceres-01.txt
```

## Results
(Quantized models are trained from scratch.)


| Model      | Top@1 Accuracy |
| --------- | -----:|
| FP32  | 76.00 |
| W4A4     |  76.20  |
| W2A2      |    74.02 |
| W2A2 (DoreFa+PACT)      |    74.81 |

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)