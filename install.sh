pip install -r requirements.txt
python -c "from torchvision.datasets import MNIST; [MNIST('data', train, download=True) for train in (True, False)]"
