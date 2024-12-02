{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {
    "id": "GJXaAC-tA6mH"
   },
   "outputs": [],
   "source": [
    "# AI Learns the Numbers Second Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lo-tPfhFBA4w"
   },
   "source": [
    "## Initial Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Assuming you've collected the loss and accuracy values during training\n",
    "results = {\n",
    "    \"train_loss\": train_loss_values,\n",
    "    \"train_acc\": train_acc_values,\n",
    "    \"test_loss\": test_loss_values,\n",
    "    \"test_acc\": test_acc_values\n",
    "}\n",
    "\n",
    "plot_loss_curves(results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_acc = 0\n",
    "model.eval()\n",
    "with torch.inference_mode():\n",
    "    for batch, (X_test, y_test) in enumerate(test_dataloader):\n",
    "        X_test, y_test = X_test.to(device), y_test.to(device)\n",
    "        test_pred = model(X_test)\n",
    "        test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))\n",
    "\n",
    "    # Average accuracy\n",
    "    test_acc /= len(test_dataloader)\n",
    "print(f\"Test Accuracy: {test_acc:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_acc = 0\n",
    "model.eval()\n",
    "with torch.inference_mode():\n",
    "    for batch, (X_test, y_test) in enumerate(test_dataloader):\n",
    "        X_test, y_test = X_test.to(device), y_test.to(device)\n",
    "        test_pred = model(X_test)\n",
    "        test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))\n",
    "\n",
    "    # Average accuracy\n",
    "    test_acc /= len(test_dataloader)\n",
    "print(f\"Test Accuracy: {test_acc:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_weights_path = \"modelWeights/trained_model.pth\"\n",
    "\n",
    "# Check if the model weights exist, then load them\n",
    "if os.path.exists(model_weights_path):\n",
    "    model.load_state_dict(torch.load(model_weights_path))\n",
    "    print(f\"[INFO] Loaded pre-trained weights from {model_weights_path}\")\n",
    "else:\n",
    "    print(\"[INFO] No pre-trained weights found, training from scratch.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {
    "id": "yqkqDRDVJdwr"
   },
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import torch\n",
    "from torch import nn\n",
    "from torch.utils.data import DataLoader\n",
    "import torchvision\n",
    "from torchvision import datasets\n",
    "from torchvision import transforms\n",
    "from torchvision.transforms import ToTensor\n",
    "import matplotlib.pyplot as plt\n",
    "import random\n",
    "import requests\n",
    "from pathlib import Path\n",
    "from tqdm.auto import tqdm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 431,
     "referenced_widgets": [
      "b343c3207d4940acb8882ac5e587e042",
      "84d3b59879694092838841f4d32386b1",
      "87084b3a21174948ae6b46d0b385675a",
      "caecbc38daf242a28da56061907c3832",
      "8a2a44438bcc4e0f9d0fcb54804f37fb",
      "ffd326cbadbc4994ae8ef210b350f4b1",
      "581655a7a9a5426ba0caaf33648b3c09",
      "befc5d94a31646e6a84a4da5dc14d49b",
      "9820c245020e4b85914ee9f4eb043ba0",
      "88bb3f39ea85452a8c23cc63c9b7d589",
      "b283a887a32c49f68290be6616025d3a",
      "e78b546c4fcd4c6d9868a734dd09e495",
      "91334d71d68842f8ac8ac4fd030b1e6e",
      "0ae23734c1ac422da72ee97d8cd83bac",
      "ec84b44f4ecd4788b73da8041b7528cb",
      "910702cbd449466f9a0ef7271b0e5f94",
      "3f493d8dcddc48f98e44bc939980ddf7",
      "b42ddf2713364a11bba653b17269ea2d",
      "1e9cf9f7cafa4ee2a7fc2a2432ae760f",
      "5cce2e0704c841169bb468498ca20d01",
      "e3d4a4cb76fc42a4b3747a82fab1c02c",
      "ad4844c2efd54a9585bb76e553fa53a5",
      "d6d948bc97094d9fb135ded270fabfa4",
      "657de865fff34eceac15774ba927fbcf",
      "71f70fe1622b45cebf237bb4d288a909",
      "d6d722f7b1554b40bef8441821653e8f",
      "545a19ff5ea8499597f894894cd20c3f",
      "c1ab528b7de847f98d36bce1f22c83b9",
      "b70871af021b4b77aaca52471e89dd9e",
      "c441e9e5a6034931a17df38e0593bcc9",
      "7abafbe26af340ac93519a610915432f",
      "edbf92155a6644d89aa4c11e2ea156d9",
      "66e5bf5e46f940b0b85aa7ceb23427ca",
      "aa17afe7fb0c40a2b8aee951a002cbc3",
      "5152cdbdd4c945c3b2ebe6f090f82a02",
      "283196f7580c45dab33f21dba2ccb7af",
      "f0fd168fa1084ab2a372e8aa8e41d8e7",
      "3c83eca20c3046618e01064a4f81d55c",
      "da9baef008544f30a7d9a6923dbacfcc",
      "d5dfab5251dd46f6b7e7ef53fbbe2a38",
      "fe6ef9b4b05d418c8e114d8e914df2bf",
      "7a5aee224a934477ae58ad93c07ac0f3",
      "7917850286ad4d89bc4197248407a610",
      "2a1df6551f214daab9833625e2ac000f"
     ]
    },
    "id": "TQjPudpELsV9",
    "outputId": "cbd1b9ac-77c4-40a1-aacd-9daa6da39f68"
   },
   "outputs": [],
   "source": [
    "# Creating and downloading the MNIST dataset\n",
    "train_data = datasets.MNIST(\n",
    "    root=\"data\",\n",
    "    train=True,\n",
    "    download=True,\n",
    "    transform=ToTensor(),\n",
    "    target_transform=None\n",
    ")\n",
    "\n",
    "test_data = datasets.MNIST(\n",
    "    root=\"data\",\n",
    "    train=False,\n",
    "    download=True,\n",
    "    transform=ToTensor(),\n",
    "    target_transform=None\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 35
    },
    "id": "AoyHLDWJGD1j",
    "outputId": "cb1727a7-2285-4633-ab11-626ddc5cd705"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'cuda'"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Write Device Agnostic Code\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "device"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TCUyd0r_BE51"
   },
   "source": [
    "## Dataset Characteristics\n",
    "This is to explore the MNIST dataset to see what type of data we're going to train our model on."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "VCZMvf6SMKcf",
    "outputId": "c2019688-e437-42c8-d9dc-eb6ff2e5bb08"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 10000)"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Size of Dataset\n",
    "len(train_data), len(test_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Vrj-ZKJgMSv_",
    "outputId": "50fb50ff-14ef-4831-f06b-de1e9f5d598e"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['0 - zero',\n",
       " '1 - one',\n",
       " '2 - two',\n",
       " '3 - three',\n",
       " '4 - four',\n",
       " '5 - five',\n",
       " '6 - six',\n",
       " '7 - seven',\n",
       " '8 - eight',\n",
       " '9 - nine']"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Dataset Labels\n",
    "train_data.classes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 299
    },
    "id": "8BOXmZPWOC56",
    "outputId": "5860a1be-ada0-498f-cdd5-4e2301af4b72"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Number 8')"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaEAAAGxCAYAAADLfglZAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAg0UlEQVR4nO3de3BU5f3H8c8KYROYZNsAuUGMEVEpcVBAgagQqKRExUKwBpEavDAqlw5GBg3QknH8EaUD2g4Vq9gAFZSqiFhAiAMJWkoNCELRIgzhUiGmRMiGgAvI8/uDYcc1ETnLbp5c3q+ZM5Nz+e755uFMPjx7OesyxhgBAGDBZbYbAAC0XIQQAMAaQggAYA0hBACwhhACAFhDCAEArCGEAADWEEIAAGsIIQCANYQQmqUFCxbI5XIpMjJS+/fvr7M/IyNDaWlpFjqTSkpK5HK59NZbb1k5/3kVFRWaMGGCrrzySkVFRSklJUUPPfSQDhw4YLUvtCyEEJo1n8+n6dOn226j0fH5fOrfv7+WLl2qyZMna/Xq1Zo6dapWrlyp9PR01dTU2G4RLQQhhGZtyJAhWrJkiT799FPbrTS4kydP6oduDfnhhx9q9+7devbZZzVu3DhlZGTokUce0fPPP68vv/xSH3zwQQN3i5aKEEKzNmXKFLVv315PPvnkBY/bt2+fXC6XFixYUGefy+VSQUGBf72goEAul0vbt2/Xr371K3k8HsXGxiovL09nzpzRrl27NGTIEEVHR+uKK67QrFmz6j3nN998o7y8PCUkJCgqKkoDBgzQ1q1b6xy3efNm3XXXXYqNjVVkZKRuuOEG/e1vfws45vzTj2vXrtWDDz6ojh07qm3btvL5fPWeOyIiQpLk8XgCtv/kJz+RJEVGRv7QUAEhRQihWYuOjtb06dO1Zs0arVu3LqSPfc8996hHjx56++23NXbsWD3//PN6/PHHNWzYMN1xxx165513NGjQID355JNatmxZnfqpU6dq7969mj9/vubPn69Dhw4pIyNDe/fu9R+zfv163XzzzTp27Jheeuklvfvuu7r++uuVk5NTb2A++OCDioiI0F//+le99dZb/rD5vptvvlm9evVSQUGBysrKdPz4cX3yySeaOnWqevbsqdtuuy1k4wRckAGaoaKiIiPJlJWVGZ/PZ6688krTu3dvc/bsWWOMMQMGDDDdu3f3H19eXm4kmaKiojqPJcnMmDHDvz5jxgwjycyePTvguOuvv95IMsuWLfNvO336tOnYsaPJzs72b1u/fr2RZHr27Onvxxhj9u3bZyIiIszDDz/s33bttdeaG264wZw+fTrgXHfeeadJTEw03377bcDve//991/0GHm9XjN06FAjyb9kZGSYqqqqi34M4FIxE0Kz16ZNGz3zzDPavHlznaexLsWdd94ZsN6tWze5XC5lZWX5t7Vu3VpXXXVVve/QGzVqlFwul389JSVF6enpWr9+vSRpz549+s9//qP77rtPknTmzBn/cvvtt+vw4cPatWtXwGOOGDHiono/ffq0cnJytG3bNr3yyivasGGDFi5cqC+//FKDBw9WdXX1xQ0CcIkIIbQII0eOVM+ePTVt2jSdPn06JI8ZGxsbsN6mTRu1bdu2zuspbdq00TfffFOnPiEhod5tVVVVkqSvvvpKkjR58mRFREQELOPGjZMkHTlyJKA+MTHxonp/9dVXtXr1ai1btkwPP/ywbr31Vt1///16//339cknn+iFF164qMcBLlVr2w0ADcHlcum5557T4MGD9fLLL9fZfz44vv9C/vlACIeKiop6t7Vv316S1KFDB0lSfn6+srOz632Ma665JmD9uzOrC9m2bZtatWqlnj17Bmy/8sor1b59e/373/++qMcBLhUzIbQYt912mwYPHqynn35ax48fD9gXHx+vyMhIbd++PWD7u+++G7Z+Xn/99YC3UO/fv18bN25URkaGpHMB07VrV3366afq3bt3vUt0dHRQ505KStK3336rsrKygO1ffPGFqqqq1Llz56B/L8AJZkJoUZ577jn16tVLlZWV6t69u3+7y+XS6NGj9Ze//EVdunRRjx499PHHH2vJkiVh66WyslLDhw/X2LFjVV1drRkzZigyMlL5+fn+Y/785z8rKytLv/jFLzRmzBh16tRJX3/9tT7//HN98sknevPNN4M69wMPPKDnn39eI0aM0PTp03XNNddo7969mjlzptq1a6dHH300VL8mcEGEEFqUG264Qffee2+94TJ79mxJ0qxZs3T8+HENGjRIf//733XFFVeEpZeZM2eqrKxMDzzwgLxer2666Sa98cYb6tKli/+YgQMH6uOPP9b//d//adKkSTp69Kjat2+vn/3sZ7rnnnuCPndycrLKysr09NNP67nnntPh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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot One Image\n",
    "CHANGE_THIS_VARIABLE_TO_SEE_ANOTHER_NUMBER = 55\n",
    "image, label = train_data[CHANGE_THIS_VARIABLE_TO_SEE_ANOTHER_NUMBER]\n",
    "plt.imshow(image.squeeze(), cmap=\"gray\")\n",
    "plt.title(\"Number \"+str(label))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 526
    },
    "id": "nvh5uanePR-j",
    "outputId": "402d1beb-bf18-4c20-d5eb-bc9e7e08bd2c"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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WbFihXz88cfSvXt3efjhh6VTp06ycuVK2bZtm7z22mu21oHv0MNpca84+NDDaWW093Uh/i7AEy6XSyZOnCj16tWTt99+W339WkP9/QGA1xrFCUePHjVm105+0dHRIiIyZMgQadGihXGNUqVKpfnz/+4838z999/v9muWZfFpjABFH9vzwAMPiIjIL7/84tE68D56WGvUqJEkJibK3r17JTk5WUqWLHn9t5oeeeQR2+vAN+hhezgPBzb6OK1du3ZJYmKi5MqVS32tVq1aEhkZKWfOnLG1FnyDHnZv2bJlcuzYMRkxYkS658J36GH36OHgQR+ntX37dsmcObNUrFgxTV60aFHJnTu37Nq1y9Y68B16OC3uFQcfejitjPa+Lqg340RE6tatK/Xq1ZMxY8ZIbGxsmq/FxMRIeHi47Ny5M03u5G90z5s3T/r373+9oRITE+Xbb7+VTp06icjVxitRooTs2LFDxo8f79VjN2rUSLJmzSorVqyQfv36Xc+3bdsmR48elSpVqnj1ePAe+vjWrn30uXjx4j45HtKHHtZcLpeUKFFCRET++usvmTZtmpQvX57NuABFD98a5+HARx/fMH/+fPVbnStXrpSJEyfKzJkzpWzZsl49HryDHjaLj4+X8PDw6//0DwIXPWxGDwcX+viGAgUKSEpKimzZskUefPDB6/kvv/wiJ0+elLvvvturx4N30MM3cK84ONHDN2S093VBvxknIjJx4kSpVKmSHDt2LM3/AJfLJR07dpR3331XihUrJnFxcbJ582b58MMPHavl2LFj0rx5c+nRo4ecPXtWRo0aJeHh4TJkyJDrY9566y1p1KiRNGjQQDp37iwFCxaUU6dOyU8//STbtm2Tjz766LaOnTNnThkzZowMHDhQOnfuLO3bt5ejR4/KiBEjpFChQtc/ForARB/fWHfDhg1Sv359iY2NlQsXLsiGDRvkjTfekGrVqsljjz3mrW8TXkYP3/D8889LzZo1JXfu3LJv3z55/fXX5dChQ7J+/XpvfHtwCD18Y13Ow8GLPr7KdGNh//79IiJSqVIl/pm0AEYPp3X48GFZuXKltG3b1vgbwQg89HBa9HBwoo+v6tKli0ydOlVatmwpw4cPl1KlSsm+fftk/Pjxctddd8kzzzzjrW8TXkYPX8W94uBFD1+V0d7XZYjNuAoVKkj79u2NTXftGT2T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ucrKlCnj0XF++eUXldWvX9849siRIyr74YcfbI0z3fsWEWnSpImtmoIBn4wDAAAAAAAAAAAAHMJmHAAAAAAAAAAAAOAQNuMAAAAAAAAAAAAAh7AZBwAAAAAAAAAAADhEP9E6iJgeNu9yubx+nEyZ2LOE75QsWdKYz507V2WWZans9OnTKhs6dKjnhXmZqXYRkWzZsqksNDRUZUlJSV6vCd5TvHhxY/7yyy+rrF27dipz1x8me/futV+YB/7zn//45DgAEAhM19Smh2yLmM/DkZGRKlu3bp1xflxcXPqK+x9z5swx5lWrVr3tNRGcoqKiVPbOO++o7PHHH/foOO6uUZo1a6aytWvXquy3335T2Z49ezyqCYGhaNGitsf+9NNPKktISPBmOT41evRoY/7ee++p7M0331TZ999/b5y/a9cuj+pCYNq3b5+/SwACVtmyZVWWL18+lZmuRxYtWuRITfC/2NhYY7569WqVlShRwtaaqampxnzZsmUqa968ua013dm/f7/KDh48qLICBQp4dJxgwC4TAAAAAAAAAAAA4BA24wAAAAAAAAAAAACHsBkHAAAAAAAAAAAAOITNOAAAAAAAAAAAAMAhbMYBAAAAAAAAAAAADgnxdwGeePnll1X27LPPev04qampxnzatGkq69q1q8o6duxonJ+YmOhZYciQGjdubMyjoqJUZurNcePGeb0mT917770ezbcsy0uVwFf69OljzNu2basy0//flJQUlU2ePNm45tKlS9NZ3e35+uuvPZqfI0cOlf3111/GsZcvX/boWACQHrGxsSoznZtPnDhhnG86Zy1btkxlcXFxxvme/JyPiYkx5vny5VPZ0aNHb/s4CHyjRo1S2eOPP+7Rmnv27FGZu+taU3+bsv/85z+2joPg83//93+2x2a0ewEffPCBMf/HP/6hsrJly6osZ86c3i4JAax69er+LgEIWG+//bbKQkL07ftJkyap7NKlS47UBP/r27evMS9evLjK7L63GjNmjDE37bd4qmTJkirzpPZgxifjAAAAAAAAAAAAAIewGQcAAAAAAAAAAAA4hM04AAAAAAAAAAAAwCFsxgEAAAAAAAAAAAAO0U+ADCInT55U2cSJE41j+/fvr7LQ0FCPjp81a1aVVa1aVWWmB9iLiBw8eFBl3bp1U9mRI0duozoEg6JFi6rsxRdftD3/+PHjKps6dapHNXkqV65cKqtTp47t+X/++afKkpOTPaoJzurXr5/KOnXq5NGaQ4YMUdkrr7zi0ZqlSpXyaP6ECRNUFhMTYxwbGxurshw5cqjs7NmzxvkPPPBAOqsDgNvXvHlzj+Z3795dZdWqVVOZ6We8iEjHjh1VVrFiRZWNHDlSZUePHjWu6S5HxmW6hh41apRHa168eFFlpveAIiL16tVT2VtvvaUy03XTBx98YFwzKSnpViUigLRv315lLpfLOHbx4sVOlxNUqlevbsy//vprH1cCAP6VM2dOW+O++uorZwuB35QsWVJl7dq182hN0/u9zz77zKM10yMqKspWdifgk3EAAAAAAAAAAACAQ9iMAwAAAAAAAAAAABzCZhwAAAAAAAAAAADgEDbjAAAAAAAAAAAAAIeE+LsAT1iWpbJhw4YZx+7Zs0dlgwYNUtm9997reWE21yxTpozKxo8fr7KJEyeqzPT9IPiYHlSdP39+2/N79+7tzXK84vXXX1dZoUKFbM/fuHGjN8uBl5keJjx27FiVRUREeHScsmXLqmz37t3GscWLF7e1ZkiIZz/yevbs6dH8CxcuqGzTpk0erQkA6RUaGmorMzGdm0VExowZY2t+o0aNjPm3336rsnvuucfWmjNmzLA1DhnfX3/9ZStz4jgiIn/++afKXC6Xynbt2qWypKQk45q9evVS2Zw5c1R26dIl43z41sKFC1X29NNPG8du2LDB6XKCSpYsWfxdAgD41FNPPWXM8+XLp7KjR4+qbPXq1V6vCYFhxIgRKjP1hYh5b2T58uUq+/zzzz0vzAOVKlXyaP6sWbO8VIn/8ck4AAAAAAAAAAAAwCFsxgEAAAAAAAAAAAAOYTMOAAAAAAAAAAAAcAibcQAAAAAAAAAAAIBD2IwDAAAAAAAAAAAAHBLi7wJ8Ze7cubYyk3Xr1hnzhx9+2JOSJFMmvRfaqVMnlTVq1EhlDRo0MK65Y8cOj2qCb7Vs2dL22F27dqls2bJl3iwn3caMGaOydu3aqcyyLJWdPn3auOakSZM8LwyOeeyxx1QWERHh9eM89dRTXl/TU6bz608//WQc+/3336ts9erVKtu5c6fnheGOZ7pOKF++vMq2b9/ufDEIeJUqVVJZhQoVbM11uVy286NHj6rs22+/Nc6PjIxU2QsvvKCy5ORklf3xxx/GNREY8ubNa8zj4+NVNn36dJXt2bPHOP/ixYsqO3bsWDqruz2m93AiItmzZ1dZjhw5bGXumK7177//fpV99913tteEc1JSUmyPrVy5ssr++9//erOcoLJ27Vp/lwAAPjVjxgxjbrp/NnToUKfLQQCpUaOGyty9D9u0aZPKTPftfCk2NlZlzz//vMpM39P69euNa/7555+eFxYg+GQcAAAAAAAAAAAA4BA24wAAAAAAAAAAAACHsBkHAAAAAAAAAAAAOITNOAAAAAAAAAAAAMAhIf4uIBi4e/DhI488orK3335bZXny5DHOT01NVZnpQZ3R0dEqW7VqlXHNfPnyGXMEpvDwcNtjd+zYobLk5GRvliMiIpkzZ1aZu4fFDh8+XGWmHjaZPHmyMb906ZKt+fCP77//XmWnT59W2V9//WWcf/ny5ds+9po1a4x5lixZbM2vXbu2MS9YsKCt+fXr11fZ8ePHbc0FnHTy5EmVZaQHHCM4LVy4UGXZsmUzjn3//fdVds8996js2LFjKlu9evVtVAdfGT16tDFv3Lixrcyds2fPqiwxMVFlGzZsUNlvv/1mXHPt2rW2jt23b19j3rBhQ5UtXbpUZU8//bSt44iIHDx40FaGwHDq1CnbY1u2bKmyBQsWeLMcAECAiIiIsJWJmK933d0DRvCrUqWKynLlyqUyd/daly1b5vWaPDVx4kSVFS9eXGWm7+nll192pKZAwifjAAAAAAAAAAAAAIewGQcAAAAAAAAAAAA4hM04AAAAAAAAAAAAwCFsxgEAAAAAAAAAAAAOCfF3AcHA9IBwEZHPPvtMZU2bNlXZ8OHDjfObNWt22zVFR0cb8169eqnszTffvO3jwHtMD+CsWLGi7fnbt2/3YjVXmWqKj49X2WOPPebRcT766COVzZo1y6M14R+7du1SWfXq1VV2/Phx4/yTJ096vSa7tmzZYswLFixoa/5ff/3lzXIAr8maNavKwsLC/FAJcEPlypVV9umnnxrH1qpVS2XJyckqe/rppz2uC76VPXt222P3799vK3PH5XKprFKlSip79tlnba9p4u56ok2bNipLSEjw6FgILqtXr1bZsGHDjGPj4uJUFh4errLLly97XliASUpKUtmlS5f8UAkA+MagQYNsj33ppZdUduTIEW+WgwBSp04dlUVERKjs2LFjxvlvv/2212uya+TIkca8bdu2KrMsS2WHDh1S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iRX20UcfRVnr1q2j7Ouvv46y8q7FkLJkyZJkfscdd0TZPffck3c5rOHatGmTzK+77rooe//996Nsl112ibJOnTol16xVq/J/v++zzz6Lsr322qvS96HmSF0jjz766MzzX3jhhSibMWNGERVVzAEHHFBle0ExmjdvnszvvPPOKOvRo0eUTZw4sdJroupNnz49yl599dUo69atW3L+wQcfXOk1VacOHToUNf+xxx6rpEpYE6X667TTTkuOnTlzZpTdfffdlV4TpaVjx47J/JBDDomyunXrRtno0aOj7PDDD0+umXrvi8rjk3EAAAAAAACQE4dxAAAAAAAAkBOHcQAAAAAAAJATh3EAAAAAAACQk9rVXQCr2njjjaPswQcfjLIuXbok599+++2VXhNARc2aNatC+X/r3LlzlK233npF1VSeH/zgB1GWerjtZ599FmUvvfRSLjVR/ZYtW5bMUw9DzmrFihXJfNGiRVFWv3791d4nhBCuuOKKKGvVqlXm+StXroyyYcOGFVUTVatTp07J/KCDDqriSr5f6vp66qmnRtmSJUuqohyqSeq6O2LEiKovZDVtvfXWqz23Xr16yXzTTTeNsrfeemu192HNMG/evGT+4osvRlnXrl0zr9u8efMoO/HEE6Ns4sSJmdektNx6661R1q1bt8zz77zzzig7+uiji6qpqsyYMaOo+SeccEIyf/bZZ4talzVD//79o6xOnTrJsa+//nqUffXVV5VeE6XlmmuuSea77rprlL333ntRduSRR0bZ4sWLiy+MCvPJOAAAAAAAAMiJwzgAAAAAAADIicM4AAAAAAAAyInDOAAAAAAAAMiJwzgAAAAAAADISe3qLqAYixYtirJPPvkkOXbDDTeMspYtW0ZZ48aNo2zBggWrUd33a9++fZSdddZZUdalS5coGzVqVHLNc889t/jCIKMLLrggyrbbbrsoa9euXeY1W7duHWUNGzaMsoULF2Zek9Lzr3/9q8r2Sl134eSTT07mI0aMWO01U9eyEEJ4+OGHV3vNvEyYMCHKzjjjjKovhEzq168fZTXx2jZ+/Phkfs4550TZlClT8i4HVsvGG2+czNdZZ50oW7lyZZR99NFHUZZ6XRpC+nvjmGOO+b4SqSaNGjWKsmOPPbbS92nSpEky79q1a5TVqhX//nWqL0NI9+Z1111XseIoaVOnTo2y2bNn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KyvLsw7WMoVCocr31MNUpuro4RD0MZXLtZhSp4cpde4nWBO4FlPq9DClzv0EawLXYkpdlh72yTgAAAAAAADIicM4AAAAAAAAyInDOAAAAAAAAMiJwzgAAAAAAADIicM4AAAAAAAAyInDOAAAAAAAAMiJwzgAAAAAAADIicM4AAAAAAAAyInDOAAAAAAAAMiJwzgAAAAAAADIicM4AAAAAAAAyInDOAAAAAAAAMiJwzgAAAAAAADIicM4AAAAAAAAyInDOAAAAAAAAMiJwzgAAAAAAADIicM4AAAAAAAAyInDOAAAAAAAAMiJwzgAAAAAAADIicM4AAAAAAAAyEnt6i4AWH1NmjSJskMPPTTK9tlnn+T8VN6oUaMoKysrS85/7bXXouz3v/99lA0bNizKVq5cmVwTAKhaf/7zn6Nszz33LGrNsWPHRln37t2LWhOqWtu2baNs9OjRybHbbLNNpvnvv/9+8YUBAEBC7drxcc+2224bZanXayGE0LRp00z71KqV/oxX6v3ep556KsoGDx4cZa+++mqmvUuZT8YBAAAAAABAThzGAQAAAAAAQE4cxgEAAAAAAEBOHMYBAAAAAABATsoKhUIh08CysrxrqRGOOuqozGOnTp2aKdt1112T84888sgoO/roozPvn1XqgYqffvppcuyAAQOi7JNPPomyMWPGFFVTxrarVKXSw+XVueOOO0ZZ6gGY6667bpSlHp5ZnjfeeCPKyuuXfffdN8pSDwp99tlno6xnz57JNRcvXvw9FdYM1dHDIZROH5eSJk2aRNkFF1wQZU8++WSUvfLKK8k1ly1bVnxhVcC1mFKnh8u32WabJfMnnngiyrbYYosoW2eddSq9poULFybzl19+Ocp69+6deX4pcz9Rsx144IFRNmrUqMzzN9100yh7//33i6qpJnItptTp4fxstNFGUVbe+149evSIss6dO2fea/z48VH2+OOPR9l9990XZeW951Eq3E+wJnAtrri6detG2bXXXhtlJ554YpSV9/7rvHnzMu1d3uvFDTbYINP8zz77LMp+8YtfJMdOmTIl05rVLUsP+2QcAAAAAAAA5MRhHAAAAAAAAOTEYRwAAAAAAADkxGEcAAAAAAAA5MRhHAAAAAAAAOSkrFAoFDINLCvLu5ZcdevWLcouuOCCKNtjjz2ibOXKlck1f/e730XZs88+G2XD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Dz/8MMpOOeWU5Pwvv/wyylIPQ049rL48bdq0ibIpU6Zkng8pJ5xwQjLfe++9M81ftGhRlJ199tnJsV988UX2wqhSWR/W/vjjjyfza665phKrqX4zZsxI5qnXBEOGDMm8btavM6WnVq3076Ouu+66VVwJVI/jjjsumR988MGZ5j/33HNRNnDgwKJqomZ44YUXouzUU09Njr311lujbP/994+yhx9+OPP+DRo0iLLU+2bl3RN36NAhylKv4ZYsWRJll156aXLN0aNHR9m0adOSY6EynHbaaZnGff3111GWer0HIYRQp06dKDvjjDOSYy+88MIoa9iwYZR9+umnyfndu3evWHFrGZ+MAwAAAAAAgJw4jAMAAAAAAICcOIwDAAAAAACAnDiMAwAAAAAAgJzUru4CKtsll1ySzD/88MMou+OOO6LsrrvuirJ99903uWarVq2irHXr1lF25JFHJuenHgzfv3//5FjIasyYMZmyivj973+fzMeNGxdlV199dZT16tUrynbeeefkmhMmTKhgdVTEe++9V637b7HFFqs9d9ttt03mBxxwwGqvWZ7Ug74rom/fvlE2atSootZk7dK5c+coO/nkk4ta88knn4yyKVOmFLUmVS/r9anY61ipGzt2bJQNHjw48/y1/etH+RYuXJjMX3rppSquBL7fxhtvHGW33nprcmzquvfmm29G2THHHBNly5cvX43qKAX33ntvMm/fvn2UPfjgg1FWq1b8GYDFixcn1xw4cGCUpe6Jy/OXv/wlyi699NIoS90jQE2Rei8h5YMPPoiySZMmVXY5rCGOOOKIKLv88suTYxctWhRljz76aJQdeuihxRe2FvLJOAAAAAAAAMiJwzgAAAAAAADIicM4AAAAAAAAyInDOAAAAAAAAMiJwzgAAAAAAADISe3qLqCq3HHHHZnG9e/fP8r22muv5NiHHnooypo0aZK5pn333TfK7r///ijr27dv5jWhKr311ltRds0110TZ/vvvH2VnnHFGcs0JEyYUXRfl++c//xll3bt3j7I//elPRe2T6o0QQliyZEmUbbvttpnWLCsrS+aFQiHT2NS4vAwfPrzK9qL0rbfeelF21113RVnTpk0zrzlnzpwoO/rooytUF6WtTZs2ybxFixZRluqXUjd16tQoGzt2bJSl7sdDSH9fnnLKKVF20003Vbw4Storr7ySzJ977rkqrgRWtfHGG0fZmDFjilrzkUceibJZs2YVtSalZfHixcn8kksuibIePXpEWeo9roq8LkuNTb1vF0IIf/jDHzKvC9Wtffv2yXz99dev4kpY05x//vlR9utf/zrKFixYkJzfu3fvKBs3blzxhRFC8Mk4AAAAAAAAyI3DOAAAAAAAAMiJwzgAAAAAAADIicM4AAAAAAAAyEnt6i6gFJT3MO7UAw3/53/+J8qOPPLI5PxWrVpF2T777BNle+yxR5S9+OKLyTWhVDRs2LC6S1grrVixIspS17iNNtoo85qpBw+/8847ybFffvlllO22225RNmPGjChLXR9DCOGwww6Lss6dOyfHVpU5c+ZU6/7UTHXr1k3mZ511VpSV90DvlM8++yzKDjzwwChbsmRJ5jUpfZ06dUrmDzzwQJT17ds3ylJ9VUq22mqrKOvSpUvm+WVlZVG23nrrFVUTNcPxxx9f1PwbbrihkiqB1bPBBhsk8zvuuCPKUvcTtWqlfyf72WefjbKrr766gtWxtpg3b16UTZo0Kco23XTTKCsUCsk1Fy1aFGV77713lE2ZMiVDhVCz/fCHP0zmWd8rGzp0aGWWQ4nq2rVrlJ1xxhlRlvrZ36dPn+Sa48aNK7YsvoNPxgEAAAAAAEBOHMYBAAAAAABAThzGAQAAAAAAQE4cxgEAAAAAAEBOald3AaXsxRdfjLLUw96PPPLIzGs2adIkyvbYY49Me0NVS/Xrsccem2nuZ599VtnlsJqWL18eZbNnz848vyJjU5566qlM4958881k/rvf/S7KynswfUq3bt1Wu6bynH/++VHWt2/fKPv666+T88t7qDmlbf/990/mF110Uab55fXFsGHDomzSpEnZC6NGmjFjRjJfsWJFlK2zzjqZ191zzz2jLHWvevfdd0dZTfzZ3axZs2T+m9/8JsrWW2+9zOu+/vrrUZb6mlCzbbnlllHWq1evXPbaaaedomzatGlRtmDBglz2Z+1xzTXXJPO99947ylL3Ds8++2xy/qGHHhplixcvrmB1rC3WX3/9KNthhx2irCKvaxYuXBhlEydOrFhhUAO1a9cuyn7/+98nx6beV547d26UPfbYY8UXRsnYbLPNkvmDDz4YZc2bN4+y66+/PsrKux8gXz4ZBwAAAAAAADlxGAcAAAAAAAA5cRgHAAAAAAAAOXEYBwAAAAAAADmpXd0FADXfJptsksyvu+66KOvZs2eUvfTSS1F2yimnFFsWa5mVK1dmHrtixYrMY8ePHx9lqYcmV+Th4wcddFCUvfzyy1H26quvJucfd9xxmfeiZtpll12i7L777itqzXvvvTeZX3jhhUWtS830wAMPJPNBgwZF2VZbbVXUXldeeWWUHXHEEVFW3oPm//73v2faZ9KkScm8VatWUda2bdso69ChQ5SVdz+x7bbbZqrp66+/TuZDhw6Nso8//jjTmtQcW2+9dZRtscUWRa3ZsWPHZL7ZZptF2cyZM6Ps0ksvLWp/1ly1a8dvz4waNSrK9ttvv8xrpl6H9ejRIzl28eLFmdeF1H1C6md3yhtvvJHMU/cDsCbo06dPlJX3/ZJ63+Hmm2+Osvnz5xdfGCXjxBNPTOYtWrSIslNPPTXK7rnnnkqvqViNGzeOsi5duiTHpn7mvPvuu1E2ePDgouvKm0/GAQAAAAAAQE4cxgEAAAAAAEBOHMYBAAAAAABAThzGAQAAAAAAQE4cxgEAAAAAAEBOald3AZVt2223Teb/+te/ouyrr77KuZpv1KqVPvMsL/9vPXr0iLKbb745OXbu3LnZC6NKtWvXLsrOO++8KLvmmmuS8995553V3nvzzTdP5r/4xS+ibJdddomyffbZJzm/UaNGUfbSSy9F2YEHHhhlCxYsSK4JeenYsWMyP+igg6KsUChU+v4//elPo2zjjTeu9H2oes2bN4+yq6++Osrq1auXec333nsvyi666KKKFcYa6bLLLouyK664Ispat25d1D4/+clPouyWW24pas0nnngimW+yySZRtvXWWxe1V1bTpk1L5iNHjqyS/clXWVlZpa85dOjQzGM33XTTSt+fNdfll18eZd27d888/9lnn42yPn36RNnixYsrVhgk7LbbblGWuubedtttUTZ69Ojkmk8++WSUDR8+PMpOOumk5PylS5cmc6hu/fr1yzx23rx5UXbrrbdWYjWUojZt2mQe+49//CPK8jgDqVOnTjLv2bNnlP385z+Psh133DHKmjVrllxz3LhxUfb0009/T4U1k0/GAQAAAAAAQE4cxgEAAAAAAEBOHMYBAAAAAABAThzGAQAAAAAAQE5qV3cBla1Hjx7JfJ999omy1MNdUw+cLRQKmfdv0qRJlK1cuTLz/NTY1MNt586dm3lNqlbt2ulvq/vvvz/K2rVrF2Unnnhi5r0222yzKDvvvPOi7H/+53+S8+vWrRtl8+fPj7LHH388OX/kyJFRNmbMmChbtmxZcj5UpZYtWybzK664Isq23377KDvggAMqvaZFixZV+prkq1at+PeYfvvb30bZTjvtlHnN1DWyb9++Ufb+++9nXpM113333RdlEydOjLJTTjklOf+0006r9JqySj3MO4SK3SsX46233oqyQw45pEr2pnpU5HVcsSZPnhxlX375ZZXtT2np2LFjlJ155plRlurhl156Kblmnz59omzBggWrUR38x/rrr5/M99hjjyhL9esbb7wRZan3DEII4c4774yyY489Nsquvfba5Pxp06Ylc6hKW221VZS1aNEi8/ybb745yj7++OOiamLt0r179yjbYostKn2fgQMHJvNNNtkk0/xx48ZF2a9+9avk2AkTJmSuq6bzyTgAAAAAAADIicM4AAAAAAAAyInDOAAAAAAAAMiJwzgAAAAAAADISe3qLqCq7LDDDpnG1aoVn09W1UPlWTM0aNAgmad6MPWA44o86L127fhbOJU9//zzyfnDhw+PssceeyzKli5dmrkmqKlSD4ctz5NPPhllqe/tvfbaq6iasj7Ylprj9NNPj7K+fftmmrtw4cJkfuCBB0bZlClTKlYYa7X33nsvym688cbk2NR9bepB2euuu27xhVWy5cuXR9nixYuTY++4444ou+WWW6Js+vTpxRdGjfX5559H2fz585NjmzRpkmnN1M+BEEJ45JFHouyLL77ItCZrrh/96EfJ/IUXXoiysrKyKFuwYEGUDR48OLlmaiwUq7xrZuo9hkMPPTTKynt/JOW6666LsmOPPTbzfKgJzjvvvChr3rx5lP39739Pzr/ssssqvSZK39/+9rdk3rt37ygbMGBA3uWEENKvzUJIvw59+umno2z8+PFRtmTJkuILq+F8Mg4AAAAAAABy4jAOAAAAAAAAcuIwDgAAAAAAAHLiMA4AAAAAAABy4jAOAAAAAAAAclK7uguobDNmzEjmL774YpRtueWWUda6devKLom1zFdffZXMO3bsGGWXX355lP3iF7/IvNfChQujrH///lF2//33J+cXCoXMe8HaZNiwYVE2ZsyYKLvllluS8/fff/9Kr4mqdcYZZyTzK664ItP89957L8qGDh2aHPvCCy9kLQsymz59ejI/66yzouzaa6+Nsn79+mXeK3XvstNOO2We//LLL0fZU089FWVvv/12lI0aNSrzPqx9nnvuuSg7/PDDk2NPOumkKHv++eej7IknnkjOnz17dgWrY01Tv379KCvvZ3/Tpk2jLPXa7Kabboqyl156qeLFwWpauXJlMj/99NOjbJNNNomyQYMGRdn222+fXLN9+/aZatp3332T+bRp0zLNh8pQVlaWzBs1apRp7Jw5c5LzFy9eXFxhrJHuvffeZJ66bh5xxBFF7TVlypQou+SSS6KsvPuR1HvV/IdPxgEAAAAAAEBOHMYBAAAAAABAThzGAQAAAAAAQE4cxgEAAAAAAEBOygqppwSnBpbzYMpS1qVLlyjbfPPNo+zss89Ozt9qq62K2r9Wrfgs9NNPP42y/fbbL8pee+21ovaubhnbrlKtiT1M9amOHg5BH9c0derUSebrrbdeUet+8cUXRc3Pam28Fjdp0iTKfvWrX0XZ5Zdfnpyfqv/rr7+Osttvvz3KUg+6pzhrYw+zZnE/wZrAtfgbAwYMiLKhQ4dmnv/BBx9E2c9+9rMomzFjRoXq4vvp4crRrFmzKLvpppuirE+fPpnX/Oijj6Js3333TY6dNm1a5nXXNO4nql7Pnj2T+ahRo6Is9d/njDPOSM7/3e9+V0xZJc21mFKXpYd9Mg4AAAAAAABy4jAOAAAAAAAAcuIwDgAAAAAAAHLiMA4AAAAAAAByUlbI+HREDzSkMnkoJ6XOA5JZE6yN1+Jf//rXUXbNNdcUteZvfvObKBs8eHBRa5LN2tjDrFncT7AmcC3+xsUXXxxlF154Yeb5Bx98cJQ9/vjjxZRERnqYUud+oupNmjQpmW+//fZRlvrv8/nnnyfnd+jQIcpmz55dwepKk2sxpS5LD/tkHAAAAAAAAOTEYRwAAAAAAADkxGEcAAAAAAAA5MRhHAAAAAAAAOSkdnUXAABQVf7xj38UNf/222+Psuuvv76oNQGAtcuFF14YZY8//njVFwLAann66aeTeefOnaMs9Rr0vPPOS86fPXt2cYUBNZpPxgEAAAAAAEBOHMYBAAAAAABAThzGAQAAAAAAQE4cxgEAAAAAAEBOHMYBAAAAAABATsoKhUIh08CysrxrYS2Sse0qlR6mMlVHD4egj6lcrsWUOj1MqXM/wZrAtZhSp4cpde4nWBO4FlPqsvSwT8YBAAAAAABAThzGAQAAAAAAQE4cxgEAAAAAAEBOHMYBAAAAAABATsoK1fWUTwAAAAAAAFjD+WQcAAAAAAAA5MRhHAAAAAAAAOTEYRwAAAAAAADkxGEcAAAAAAAA5MRhHAAAAAAAAOTEYRwAAAAAAADkxGEcAAAAAAAA5MRhHAAAAAAAAOTEYRwAAAAAAADk5P8BaADUavcp2AoAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 2250x900 with 40 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Plot 40 Random Images from the Datset - Re-run Code to see new ones\n",
    "fig = plt.figure(figsize=(22.5, 9))\n",
    "rows, cols = 4, 10\n",
    "for i in range(1, rows*cols+1):\n",
    "  image, label = train_data[random.randint(0,len(train_data))]\n",
    "  fig.add_subplot(rows, cols, i)\n",
    "  plt.imshow(image.squeeze(), cmap=\"gray\")\n",
    "  plt.title(\"Number \"+str(label))\n",
    "  plt.axis(False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "A2MpV2jkRptj",
    "outputId": "5dac5272-c763-4f2f-bc82-5ae50f0a98e0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset MNIST\n",
      "    Number of datapoints: 60000\n",
      "    Root location: data\n",
      "    Split: Train\n",
      "    StandardTransform\n",
      "Transform: ToTensor()\n",
      "\n",
      "Dataset MNIST\n",
      "    Number of datapoints: 10000\n",
      "    Root location: data\n",
      "    Split: Test\n",
      "    StandardTransform\n",
      "Transform: ToTensor()\n"
     ]
    }
   ],
   "source": [
    "# Make PyTorch DataLoader\n",
    "BATCH_SIZE = 32\n",
    "\n",
    "# Dataset Used to Train the Model\n",
    "train_dataloader = DataLoader(dataset=train_data,\n",
    "                              batch_size=BATCH_SIZE,\n",
    "                              shuffle=True)\n",
    "\n",
    "# Dataset Used to Test the Model\n",
    "test_dataloader = DataLoader(dataset=test_data,\n",
    "                              batch_size=BATCH_SIZE,\n",
    "                              shuffle=False)\n",
    "print(train_dataloader.dataset)\n",
    "print(\"\")\n",
    "print(test_dataloader.dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Preparation\n",
    "# Define the transformation for data normalization\n",
    "transform = transforms.Compose([\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize((0.5,), (0.5,))\n",
    "])\n",
    "\n",
    "# Load the training set from consolidated directory\n",
    "trainset = datasets.MNIST(root='./MNISTdata', train=True, download=True, transform=transform)\n",
    "trainloader = DataLoader(trainset, batch_size=64, shuffle=True)\n",
    "\n",
    "# Load the testing set from consolidated directory\n",
    "testset = datasets.MNIST(root='./MNISTdata', train=False, download=True, transform=transform)\n",
    "testloader = DataLoader(testset, batch_size=64, shuffle=False)\n",
    "\n",
    "# Load pre-trained model weights if available\n",
    "model_weights_path = './modelWeights/trained_model.pth'\n",
    "if os.path.exists(model_weights_path):\n",
    "    model.load_state_dict(torch.load(model_weights_path))\n",
    "    print(f\"Loaded pre-trained weights from '{model_weights_path}'\")\n",
    "else:\n",
    "    print(\"No pre-trained weights found. Training the model from scratch.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Images batch shape: torch.Size([64, 1, 28, 28])\n",
      "Labels batch shape: torch.Size([64])\n"
     ]
    }
   ],
   "source": [
    "# Verify data loading\n",
    "data_iter = iter(trainloader)\n",
    "images, labels = next(data_iter)\n",
    "\n",
    "# Print the shapes of images and labels to confirm loading\n",
    "print(f\"Images batch shape: {images.shape}\")\n",
    "print(f\"Labels batch shape: {labels.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "G4xUWKKhH2cW"
   },
   "source": [
    "## Creating the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {
    "id": "j_3sjoXpTkOb"
   },
   "outputs": [],
   "source": [
    "# Make Model 2\n",
    "# The structure can be seen below (1 Flatten layer, 2 Linear Layers)\n",
    "class DumbModel(nn.Module):\n",
    "  def __init__(self,\n",
    "               input_shape: int,\n",
    "               hidden_units: int,\n",
    "               output_shape: int):\n",
    "    super().__init__()\n",
    "    self.layer_stack = nn.Sequential(\n",
    "        nn.Flatten(),\n",
    "        nn.Linear(in_features=input_shape, out_features=hidden_units),\n",
    "        nn.Linear(in_features=hidden_units, out_features=output_shape),\n",
    "    )\n",
    "  def forward(self, x):\n",
    "    return self.layer_stack(x)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "AlBlafwkVdWy",
    "outputId": "131be6ed-d1f9-4754-84fb-16ebee90696c"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DumbModel(\n",
       "  (layer_stack): Sequential(\n",
       "    (0): Flatten(start_dim=1, end_dim=-1)\n",
       "    (1): Linear(in_features=784, out_features=256, bias=True)\n",
       "    (2): Linear(in_features=256, out_features=10, bias=True)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 111,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = DumbModel(input_shape=784,\n",
    "                      hidden_units=256,\n",
    "                      output_shape=len(train_data.classes)).to(device)\n",
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "7uhJE6uUWW_m",
    "outputId": "67ca00b2-903e-4638-eeae-f1d00fccecf9"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Python script already downloaded\n"
     ]
    }
   ],
   "source": [
    "# Getting helper functions from my friend daniel bourke\n",
    "\n",
    "if Path(\"helper_functions.py\").is_file():\n",
    "  print(\"Python script already downloaded\")\n",
    "else:\n",
    "  print(\"Downloading helper_functions.py script\")\n",
    "  request = requests.get(\"https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py\")\n",
    "  with open(\"helper_functions.py\", \"wb\") as f:\n",
    "    f.write(request.content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {
    "id": "dfhAzj0hX-W5"
   },
   "outputs": [],
   "source": [
    "from helper_functions import accuracy_fn\n",
    "from helper_functions import print_train_time\n",
    "from timeit import default_timer as timer\n",
    "\n",
    "# Creating and choosing a loss function and an optimizer\n",
    "# Change the learning rate to make the network learn faster (and generally less accurate) or slower (and generally more accurate)\n",
    "LEARNING_RATE = 0.05\n",
    "\n",
    "loss_fn = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.SGD(params=model.parameters(),\n",
    "                            lr=LEARNING_RATE,)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ofDJgGdVIBE0"
   },
   "source": [
    "## Training the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "\n",
    "# Assuming `model` is already defined\n",
    "model_save_path = \"modelWeights/trained_model.pth\"\n",
    "\n",
    "# Check if weights are available and load them\n",
    "if os.path.exists(model_save_path):\n",
    "    model.load_state_dict(torch.load(model_save_path))\n",
    "    print(f\"Model weights loaded from {model_save_path}\")\n",
    "else:\n",
    "    print(\"No pre-trained weights found. Training from scratch.\")\n",
    "\n",
    "# Set the model to evaluation mode after loading the weights\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 513,
     "referenced_widgets": [
      "4213e120a9c848d7b8b538a9b8a70f2f",
      "7d934be4250f40ce9777f41685a0ae1e",
      "6f9439de138b41e98f0781475141c582",
      "99298dcffb0a40bfaba5584f92419b32",
      "2235cd55531342069c8e34d741c4a286",
      "46f96b036f574280a5345a4b9bbb2fb2",
      "79da759132e84dfe9a821524e63e0989",
      "e511e17e4ece402a81736e838d5fd539",
      "bc5c4704a74842579ff87d9b05e42bbd",
      "727ef457f40a4f07bf560fb87a67d791",
      "776154d7e2484006a8aec47636d341cc"
     ]
    },
    "id": "pmzCnU7LZUIk",
    "outputId": "a2261992-7b93-4701-8712-ba0a4c1df699"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5a8af93941e447aa8fb97a5f0755aa49",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/8 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0\n",
      "Train loss: 0.4331 | Test loss: 0.3040 | Accuracy: 91.5335\n",
      "\n",
      "Epoch: 1\n",
      "Train loss: 0.3060 | Test loss: 0.2945 | Accuracy: 91.9030\n",
      "\n",
      "Epoch: 2\n",
      "Train loss: 0.2924 | Test loss: 0.2799 | Accuracy: 92.2424\n",
      "\n",
      "Epoch: 3\n",
      "Train loss: 0.2847 | Test loss: 0.2853 | Accuracy: 92.1925\n",
      "\n",
      "Epoch: 4\n",
      "Train loss: 0.2793 | Test loss: 0.2876 | Accuracy: 92.0527\n",
      "\n",
      "Epoch: 5\n",
      "Train loss: 0.2762 | Test loss: 0.2710 | Accuracy: 92.4221\n",
      "\n",
      "Epoch: 6\n",
      "Train loss: 0.2729 | Test loss: 0.2739 | Accuracy: 92.1526\n",
      "\n",
      "Epoch: 7\n",
      "Train loss: 0.2706 | Test loss: 0.2818 | Accuracy: 91.9928\n",
      "\n",
      "\n",
      "Train time on cuda:0: 57.377 seconds\n"
     ]
    }
   ],
   "source": [
    "# Start Timer\n",
    "train_time_start_on_cpu = timer()\n",
    "\n",
    "# Open a log file\n",
    "log_file = \"training_log.txt\"\n",
    "with open(log_file, \"w\") as f:\n",
    "    # Number of Epochs - Change this to train the network for longer\n",
    "    EPOCHS = 8\n",
    "\n",
    "    # Training Loop\n",
    "    for epoch in tqdm(range(EPOCHS)):\n",
    "        print(f\"Epoch: {epoch + 1}\")\n",
    "        train_loss = 0\n",
    "\n",
    "        # Load the data in batches to avoid memory problems\n",
    "        for batch, (X, y) in enumerate(train_dataloader):\n",
    "            X, y = X.to(device), y.to(device)\n",
    "            # Set model to train\n",
    "            model.train()\n",
    "            # Perform the forward pass\n",
    "            y_pred = model(X)\n",
    "            # Calculate the loss (how bad our model is at predicting the right values)\n",
    "            loss = loss_fn(y_pred, y)\n",
    "            # Accumulate the loss\n",
    "            train_loss += loss\n",
    "            # Set the gradients of the optimizer to zero\n",
    "            optimizer.zero_grad()\n",
    "            # Perform backpropagation\n",
    "            loss.backward()\n",
    "            # Performs a single optimization step\n",
    "            optimizer.step()\n",
    "        # Calculate the loss across all the data\n",
    "        train_loss /= len(train_dataloader)\n",
    "\n",
    "        # Test the model on unseen data\n",
    "        test_loss, test_acc = 0, 0\n",
    "        model.eval()\n",
    "\n",
    "        with torch.inference_mode():\n",
    "            for batch, (X_test, y_test) in enumerate(test_dataloader):\n",
    "                X_test, y_test = X_test.to(device), y_test.to(device)\n",
    "                # Our model predicts on test data\n",
    "                test_pred = model(X_test)\n",
    "                # Calculate loss and accuracy\n",
    "                test_loss += loss_fn(test_pred, y_test)\n",
    "                test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))\n",
    "            # Calculate loss and accuracy\n",
    "            test_loss /= len(test_dataloader)\n",
    "            test_acc /= len(test_dataloader)\n",
    "\n",
    "        # Print and log training and test results for the current epoch\n",
    "        print(f\"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%\\n\")\n",
    "        f.write(f\"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\\n\")\n",
    "        f.flush()\n",
    "\n",
    "# Stop timer\n",
    "train_time_end_on_cpu = timer()\n",
    "# Calculate timer difference\n",
    "total_train_time_model = print_train_time(start=train_time_start_on_cpu,\n",
    "                                          end=train_time_end_on_cpu,\n",
    "                                          device=str(next(model.parameters()).device))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "45rRiGr0RCbn",
    "outputId": "f69afb62-044a-4622-8cf2-e973dcf1afd9"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "OrderedDict([('layer_stack.1.weight',\n",
       "              tensor([[ 0.0025, -0.0065,  0.0040,  ..., -0.0002, -0.0271, -0.0215],\n",
       "                      [-0.0036, -0.0117, -0.0288,  ..., -0.0103, -0.0274, -0.0074],\n",
       "                      [ 0.0296, -0.0230, -0.0083,  ..., -0.0148, -0.0352,  0.0274],\n",
       "                      ...,\n",
       "                      [-0.0357,  0.0235,  0.0154,  ...,  0.0063, -0.0153, -0.0192],\n",
       "                      [-0.0141, -0.0066,  0.0066,  ...,  0.0200,  0.0212,  0.0258],\n",
       "                      [ 0.0184, -0.0099,  0.0056,  ..., -0.0095,  0.0049,  0.0009]],\n",
       "                     device='cuda:0')),\n",
       "             ('layer_stack.1.bias',\n",
       "              tensor([-4.4626e-02, -1.8667e-02, -3.7431e-02,  6.2461e-02,  5.2461e-02,\n",
       "                      -4.7325e-02, -5.6779e-02, -1.5673e-01,  7.7583e-02, -2.0811e-02,\n",
       "                      -1.6119e-01,  6.0561e-02,  1.5769e-01,  1.3254e-02,  8.4099e-02,\n",
       "                       6.8345e-02,  1.4703e-01, -1.1101e-01, -9.0252e-02,  1.8453e-01,\n",
       "                      -2.7331e-02, -4.2093e-02, -1.5132e-01, -1.5960e-01, -1.5037e-02,\n",
       "                       3.2130e-02,  6.0635e-02, -1.3887e-02, -1.3217e-01,  7.2639e-02,\n",
       "                       1.3928e-01,  7.2047e-04,  1.8791e-01, -2.4067e-01, -7.8491e-02,\n",
       "                      -1.7242e-02,  1.0530e-01, -6.4560e-02, -4.4554e-02,  4.9827e-02,\n",
       "                       8.8470e-02,  5.3940e-02,  8.8286e-03, -4.9693e-03, -4.2452e-02,\n",
       "                       4.7397e-02,  5.8862e-02,  1.6636e-02, -1.3887e-01, -1.1321e-01,\n",
       "                      -9.5684e-02,  5.7161e-02,  1.3933e-02,  1.5332e-02,  1.9179e-01,\n",
       "                       2.3182e-02,  1.1852e-01, -7.6137e-02, -2.8941e-02, -4.7582e-02,\n",
       "                      -1.1957e-02,  5.5235e-02, -1.4695e-01,  5.5514e-02,  2.3712e-02,\n",
       "                      -5.8670e-02, -1.9235e-01, -7.9018e-03, -8.0602e-02, -1.4164e-01,\n",
       "                      -3.0242e-02, -1.1817e-01,  3.4651e-02, -4.7821e-02,  7.9813e-02,\n",
       "                      -4.3591e-02, -1.2627e-01,  3.8815e-02, -7.1192e-03, -1.9349e-04,\n",
       "                      -1.7209e-02,  2.8231e-02,  1.2586e-01,  6.1222e-02, -1.2360e-02,\n",
       "                      -9.8323e-03,  1.3108e-01,  2.0518e-02, -1.3513e-02,  1.2161e-01,\n",
       "                       8.2231e-02,  3.9734e-02, -5.9532e-02, -7.7050e-02,  9.8001e-02,\n",
       "                       1.4245e-01, -6.9132e-02, -2.6146e-01,  1.7750e-02, -8.9335e-02,\n",
       "                      -8.1358e-04,  1.3519e-01, -8.2057e-02,  8.4010e-02,  3.5372e-02,\n",
       "                      -2.8155e-02, -1.4688e-01, -9.6917e-02,  1.8496e-03, -1.2467e-01,\n",
       "                       1.6668e-01, -1.3395e-01, -1.5389e-03, -3.9213e-02, -1.1434e-01,\n",
       "                       8.3390e-02, -3.6775e-02, -5.0478e-02, -7.6871e-03,  1.8745e-01,\n",
       "                       2.0548e-01, -1.0151e-02, -2.2500e-01, -1.1624e-01, -2.7094e-02,\n",
       "                      -1.6105e-01, -1.0390e-01, -9.8864e-02,  6.1636e-02, -1.6739e-02,\n",
       "                      -1.4121e-01, -1.9747e-01,  6.2307e-02,  2.7312e-02,  9.5021e-02,\n",
       "                       1.7715e-02,  4.5456e-02, -1.1489e-01,  9.1958e-03,  6.3187e-04,\n",
       "                      -1.3526e-01,  4.2268e-03, -1.7036e-02, -7.1377e-02, -1.5637e-01,\n",
       "                      -1.2519e-01,  9.3813e-02,  5.6390e-02, -3.2252e-02, -4.6466e-02,\n",
       "                       3.7066e-02,  1.6471e-01,  2.9038e-02, -5.2637e-02,  1.0646e-01,\n",
       "                      -9.4284e-02, -2.6005e-02,  2.5783e-03,  1.4497e-01,  9.5588e-02,\n",
       "                      -7.8110e-02,  7.1609e-02,  1.2438e-01,  4.0537e-02, -1.3648e-01,\n",
       "                       4.6026e-02,  9.3544e-02,  1.9228e-01, -7.3233e-02,  1.0711e-02,\n",
       "                       1.2341e-02,  1.1843e-01, -5.4131e-02,  4.5515e-03, -8.1742e-02,\n",
       "                       2.8233e-02,  1.0609e-01,  5.4282e-02, -1.7390e-02, -4.6029e-02,\n",
       "                       5.7429e-02, -8.1195e-03,  2.4658e-03,  5.1790e-02, -1.3309e-01,\n",
       "                       1.4052e-01,  1.0174e-01,  6.1015e-02, -1.2198e-02, -5.1847e-02,\n",
       "                      -3.2687e-04,  6.4409e-02, -3.6522e-02,  1.8850e-01,  1.3225e-01,\n",
       "                      -6.8790e-02,  2.0155e-01, -1.3367e-01,  2.4637e-02, -1.7468e-02,\n",
       "                      -5.7022e-02,  3.6506e-02, -1.0992e-02,  5.9211e-02, -8.7614e-02,\n",
       "                      -1.1412e-01,  3.2950e-02,  5.3597e-02, -2.0634e-01,  1.8848e-01,\n",
       "                      -4.9621e-02, -2.7396e-02, -9.0700e-02, -3.4819e-02, -2.4661e-02,\n",
       "                       1.0275e-01,  4.5140e-02,  1.4830e-01,  7.0905e-02, -1.0331e-01,\n",
       "                       2.3893e-02, -3.2196e-02,  8.5558e-02,  1.2060e-01,  1.5360e-02,\n",
       "                       1.4802e-01, -4.8074e-02, -4.2885e-02, -1.4326e-01,  6.7454e-02,\n",
       "                      -6.0875e-05, -7.8969e-02,  3.6857e-03,  6.2975e-02, -7.4668e-02,\n",
       "                      -8.3461e-02, -1.4200e-01,  6.2807e-02, -7.6111e-02, -1.3991e-01,\n",
       "                       3.0550e-02, -1.9242e-01, -9.5988e-02, -1.0899e-01,  4.1218e-02,\n",
       "                      -2.0773e-02,  1.1782e-01,  1.2123e-01, -2.1060e-01,  7.0818e-02,\n",
       "                      -1.3214e-01,  9.3047e-03, -2.1961e-01, -7.4237e-02,  1.5956e-01,\n",
       "                       3.8663e-02], device='cuda:0')),\n",
       "             ('layer_stack.2.weight',\n",
       "              tensor([[ 0.0293,  0.3034, -0.0589,  ...,  0.1239, -0.1442, -0.0680],\n",
       "                      [ 0.0286, -0.1400,  0.0813,  ..., -0.0316, -0.0394,  0.1042],\n",
       "                      [ 0.1631,  0.2202,  0.1595,  ..., -0.0282,  0.1337,  0.1476],\n",
       "                      ...,\n",
       "                      [-0.0937, -0.2040, -0.2454,  ..., -0.1158,  0.0208, -0.2118],\n",
       "                      [-0.1966, -0.0952, -0.0712,  ...,  0.0111, -0.1029, -0.1352],\n",
       "                      [ 0.0620, -0.1797,  0.0743,  ...,  0.0084,  0.1091, -0.0941]],\n",
       "                     device='cuda:0')),\n",
       "             ('layer_stack.2.bias',\n",
       "              tensor([-0.3380,  0.1764,  0.1751, -0.2336, -0.0297,  0.6389, -0.1099,  0.2670,\n",
       "                      -0.6316,  0.0134], device='cuda:0'))])"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.state_dict()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [1/5], Batch [100/938], Loss: 1.2481\n",
      "Epoch [1/5], Batch [200/938], Loss: 0.5015\n",
      "Epoch [1/5], Batch [300/938], Loss: 0.3963\n",
      "Epoch [1/5], Batch [400/938], Loss: 0.3601\n",
      "Epoch [1/5], Batch [500/938], Loss: 0.3550\n",
      "Epoch [1/5], Batch [600/938], Loss: 0.3091\n",
      "Epoch [1/5], Batch [700/938], Loss: 0.3064\n",
      "Epoch [1/5], Batch [800/938], Loss: 0.2988\n",
      "Epoch [1/5], Batch [900/938], Loss: 0.2910\n",
      "Epoch [2/5], Batch [100/938], Loss: 0.2792\n",
      "Epoch [2/5], Batch [200/938], Loss: 0.2822\n",
      "Epoch [2/5], Batch [300/938], Loss: 0.2973\n",
      "Epoch [2/5], Batch [400/938], Loss: 0.3037\n",
      "Epoch [2/5], Batch [500/938], Loss: 0.3075\n",
      "Epoch [2/5], Batch [600/938], Loss: 0.3000\n",
      "Epoch [2/5], Batch [700/938], Loss: 0.2996\n",
      "Epoch [2/5], Batch [800/938], Loss: 0.2925\n",
      "Epoch [2/5], Batch [900/938], Loss: 0.2837\n",
      "Epoch [3/5], Batch [100/938], Loss: 0.2704\n",
      "Epoch [3/5], Batch [200/938], Loss: 0.2968\n",
      "Epoch [3/5], Batch [300/938], Loss: 0.3180\n",
      "Epoch [3/5], Batch [400/938], Loss: 0.2981\n",
      "Epoch [3/5], Batch [500/938], Loss: 0.2849\n",
      "Epoch [3/5], Batch [600/938], Loss: 0.2905\n",
      "Epoch [3/5], Batch [700/938], Loss: 0.2864\n",
      "Epoch [3/5], Batch [800/938], Loss: 0.2866\n",
      "Epoch [3/5], Batch [900/938], Loss: 0.2849\n",
      "Epoch [4/5], Batch [100/938], Loss: 0.2821\n",
      "Epoch [4/5], Batch [200/938], Loss: 0.2968\n",
      "Epoch [4/5], Batch [300/938], Loss: 0.2723\n",
      "Epoch [4/5], Batch [400/938], Loss: 0.2764\n",
      "Epoch [4/5], Batch [500/938], Loss: 0.2883\n",
      "Epoch [4/5], Batch [600/938], Loss: 0.2751\n",
      "Epoch [4/5], Batch [700/938], Loss: 0.2883\n",
      "Epoch [4/5], Batch [800/938], Loss: 0.3155\n",
      "Epoch [4/5], Batch [900/938], Loss: 0.2972\n",
      "Epoch [5/5], Batch [100/938], Loss: 0.2994\n",
      "Epoch [5/5], Batch [200/938], Loss: 0.3080\n",
      "Epoch [5/5], Batch [300/938], Loss: 0.2784\n",
      "Epoch [5/5], Batch [400/938], Loss: 0.2907\n",
      "Epoch [5/5], Batch [500/938], Loss: 0.2847\n",
      "Epoch [5/5], Batch [600/938], Loss: 0.2861\n",
      "Epoch [5/5], Batch [700/938], Loss: 0.2796\n",
      "Epoch [5/5], Batch [800/938], Loss: 0.2831\n",
      "Epoch [5/5], Batch [900/938], Loss: 0.2812\n",
      "Training complete.\n"
     ]
    }
   ],
   "source": [
    "# Training the Model\n",
    "# Assuming `model` is already defined under \"Creating the Model\" section\n",
    "\n",
    "# Set device to GPU if available\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "model.to(device)\n",
    "\n",
    "# Define loss function and optimizer\n",
    "loss_fn = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)\n",
    "\n",
    "# Training parameters\n",
    "epochs = 5  # Number of times we pass through the entire dataset\n",
    "\n",
    "# Training loop\n",
    "for epoch in range(epochs):\n",
    "    model.train()  # Set model to training mode\n",
    "    running_loss = 0.0\n",
    "\n",
    "    for batch_idx, (images, labels) in enumerate(trainloader):\n",
    "        # Move data to the selected device\n",
    "        images, labels = images.to(device), labels.to(device)\n",
    "\n",
    "        # Zero the parameter gradients\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        # Forward pass\n",
    "        outputs = model(images)\n",
    "        loss = loss_fn(outputs, labels)\n",
    "\n",
    "        # Backward pass and optimization\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        # Print statistics\n",
    "        running_loss += loss.item()\n",
    "        if batch_idx % 100 == 99:  # Print every 100 mini-batches\n",
    "            print(f\"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}\")\n",
    "            running_loss = 0.0\n",
    "\n",
    "print(\"Training complete.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "9a778028b1c247799611a07601e0829c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/8 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 1\n",
      "Epoch [1/8] | Train Loss: 0.2858 | Test Loss: 0.3075 | Accuracy: 91.2321\n",
      "\n",
      "Epoch: 2\n",
      "Epoch [2/8] | Train Loss: 0.2848 | Test Loss: 0.3042 | Accuracy: 91.0529\n",
      "\n",
      "Epoch: 3\n",
      "Epoch [3/8] | Train Loss: 0.2826 | Test Loss: 0.2876 | Accuracy: 91.8690\n",
      "\n",
      "Epoch: 4\n",
      "Epoch [4/8] | Train Loss: 0.2805 | Test Loss: 0.3015 | Accuracy: 91.5605\n",
      "\n",
      "Epoch: 5\n",
      "Epoch [5/8] | Train Loss: 0.2804 | Test Loss: 0.3015 | Accuracy: 91.2420\n",
      "\n",
      "Epoch: 6\n",
      "Epoch [6/8] | Train Loss: 0.2800 | Test Loss: 0.3005 | Accuracy: 91.8193\n",
      "\n",
      "Epoch: 7\n",
      "Epoch [7/8] | Train Loss: 0.2808 | Test Loss: 0.3003 | Accuracy: 91.5406\n",
      "\n",
      "Epoch: 8\n",
      "Epoch [8/8] | Train Loss: 0.2806 | Test Loss: 0.3305 | Accuracy: 90.5056\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Import necessary libraries\n",
    "import torch  # Ensure PyTorch is imported\n",
    "from torch import nn\n",
    "from torch.optim import SGD\n",
    "from tqdm.auto import tqdm\n",
    "from timeit import default_timer as timer\n",
    "\n",
    "# Set device to GPU if available\n",
    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "model.to(device)\n",
    "\n",
    "# Define loss function and optimizer\n",
    "loss_fn = nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)\n",
    "\n",
    "# Training parameters\n",
    "epochs = 8  # Number of times we pass through the entire dataset\n",
    "\n",
    "# Open a file to save the training log\n",
    "log_file_path = \"training_log.txt\"\n",
    "with open(log_file_path, \"w\") as log_file:\n",
    "\n",
    "    # Start Timer\n",
    "    train_time_start_on_cpu = timer()\n",
    "\n",
    "    # Training Loop\n",
    "for epoch in tqdm(range(epochs)):\n",
    "    print(f\"Epoch: {epoch + 1}\")\n",
    "    train_loss = 0\n",
    "\n",
    "    # Load the data in batches to avoid memory problems\n",
    "    for batch, (X, y) in enumerate(trainloader):\n",
    "        X, y = X.to(device), y.to(device)\n",
    "        # Set model to train\n",
    "        model.train()\n",
    "        # Perform the forward pass\n",
    "        y_pred = model(X)\n",
    "        # Calculate the loss (how bad our model is at predicting the right values)\n",
    "        loss = loss_fn(y_pred, y)\n",
    "        # Accumulate the loss\n",
    "        train_loss += loss.item()  # Add the scalar loss value instead of the entire tensor\n",
    "        # Set the gradients of the optimizer to zero\n",
    "        optimizer.zero_grad()\n",
    "        # Perform backpropagation\n",
    "        loss.backward()\n",
    "        # Performs a single optimization step\n",
    "        optimizer.step()\n",
    "\n",
    "    # Calculate the average train loss across all batches\n",
    "    train_loss /= len(trainloader)\n",
    "\n",
    "    # Test the model on unseen data\n",
    "    test_loss, test_acc = 0, 0\n",
    "    model.eval()\n",
    "\n",
    "    # Use inference mode to speed up and prevent gradient calculation\n",
    "    with torch.inference_mode():\n",
    "        for batch, (X_test, y_test) in enumerate(testloader):\n",
    "            X_test, y_test = X_test.to(device), y_test.to(device)\n",
    "            # Our model predicts on test data\n",
    "            test_pred = model(X_test)\n",
    "            # Calculate loss and accuracy\n",
    "            test_loss += loss_fn(test_pred, y_test).item()  # Get the scalar loss value\n",
    "            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))\n",
    "\n",
    "    # Calculate average test loss and accuracy\n",
    "    test_loss /= len(testloader)\n",
    "    test_acc /= len(testloader)\n",
    "\n",
    "    # Print and log training and test results for the current epoch\n",
    "    log_message = f\"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}\\n\"\n",
    "    print(log_message)\n",
    "    with open(log_file_path, \"a\") as log_file:\n",
    "        log_file.write(log_message)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Open a log file\n",
    "log_file = \"training_log.txt\"\n",
    "with open(log_file, \"w\") as f:\n",
    "    for epoch in range(EPOCHS):\n",
    "        # Training and evaluation code\n",
    "        f.write(f\"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\\n\")\n",
    "        f.flush()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "d0kXMV6SIEKw"
   },
   "source": [
    "## Evaluating the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "uUZ_qW6nhe6Z",
    "outputId": "413855b2-97a4-4746-c53b-1d17a2065306"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'model_name': 'DumbModel',\n",
       " 'model_loss': 0.41086331009864807,\n",
       " 'model_acc': 91.43370607028754}"
      ]
     },
     "execution_count": 118,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Evaluate Model on Unseen Data\n",
    "def eval_model(model: torch.nn.Module,\n",
    "               data_loader: torch.utils.data.DataLoader,\n",
    "               loss_fn: torch.nn.Module,\n",
    "               accuracy_fn):\n",
    "    \"\"\"Returns a dictionary containing the results of model predicting on data_loader.\n",
    "\n",
    "    Args:\n",
    "        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.\n",
    "        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.\n",
    "        loss_fn (torch.nn.Module): The loss function of model.\n",
    "        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.\n",
    "\n",
    "    Returns:\n",
    "        (dict): Results of model making predictions on data_loader.\n",
    "    \"\"\"\n",
    "    loss, acc = 0, 0\n",
    "    model.eval()\n",
    "    with torch.inference_mode():\n",
    "        for X, y in data_loader:\n",
    "            X, y = X.to(device), y.to(device)\n",
    "            # Make predictions with the model\n",
    "            y_pred = model(X)\n",
    "\n",
    "            # Accumulate the loss and accuracy values per batch\n",
    "            loss += loss_fn(y_pred, y)\n",
    "            acc += accuracy_fn(y_true=y,\n",
    "                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)\n",
    "\n",
    "        # Scale loss and acc to find the average loss/acc per batch\n",
    "        loss /= len(data_loader)\n",
    "        acc /= len(data_loader)\n",
    "\n",
    "    return {\"model_name\": model.__class__.__name__, # only works when model was created with a class\n",
    "            \"model_loss\": loss.item(),\n",
    "            \"model_acc\": acc}\n",
    "\n",
    "# Calculate model 0 results on test dataset\n",
    "model_results = eval_model(model=model, data_loader=test_dataloader,\n",
    "    loss_fn=loss_fn, accuracy_fn=accuracy_fn\n",
    ")\n",
    "model_results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "-JmQ_GZyILeP"
   },
   "source": [
    "### Making predictions of random images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {
    "id": "6DdmSau8IKuR"
   },
   "outputs": [],
   "source": [
    "def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):\n",
    "  pred_probs = []\n",
    "  model.eval()\n",
    "  with torch.inference_mode():\n",
    "      for sample in data:\n",
    "          # Prepare sample\n",
    "          sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device\n",
    "\n",
    "          # Forward pass (model outputs raw logit)\n",
    "          pred_logit = model(sample)\n",
    "\n",
    "          # Get prediction probability (logit -> prediction probability)\n",
    "          pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)\n",
    "\n",
    "          # Get pred_prob off GPU for further calculations\n",
    "          pred_probs.append(pred_prob.cpu())\n",
    "\n",
    "  # Stack the pred_probs to turn list into a tensor\n",
    "  return torch.stack(pred_probs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {
    "id": "g7ptzdKoJcdJ"
   },
   "outputs": [],
   "source": [
    "def plot_predictions(test_samples, test_labels):\n",
    "  # Plot predictions\n",
    "  plt.figure(figsize=(9, 9))\n",
    "  nrows = 3\n",
    "  ncols = 3\n",
    "  for i, sample in enumerate(test_samples):\n",
    "    # Create a subplot\n",
    "    plt.subplot(nrows, ncols, i+1)\n",
    "\n",
    "    # Plot the target image\n",
    "    plt.imshow(sample.squeeze(), cmap=\"gray\")\n",
    "\n",
    "    # Find the prediction label (in text form, e.g. \"Sandal\")\n",
    "    pred_label = train_data.classes[pred_classes[i]]\n",
    "\n",
    "    # Get the truth label (in text form, e.g. \"T-shirt\")\n",
    "    truth_label = train_data.classes[test_labels[i]]\n",
    "\n",
    "    # Create the title text of the plot\n",
    "    title_text = f\"Pred: {pred_label} | Truth: {truth_label}\"\n",
    "\n",
    "    # Check for equality and change title colour accordingly\n",
    "    if pred_label == truth_label:\n",
    "        plt.title(title_text, fontsize=10, c=\"g\") # green text if correct\n",
    "    else:\n",
    "        plt.title(title_text, fontsize=10, c=\"r\") # red text if wrong\n",
    "    plt.axis(False);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 534
    },
    "id": "oj_yGV7aIV69",
    "outputId": "d1bf54b2-d0e8-4ec2-c8fd-7380fa5f52df"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 900x900 with 9 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import random\n",
    "test_samples = []\n",
    "test_labels = []\n",
    "for sample, label in random.sample(list(test_data), k=9):\n",
    "    test_samples.append(sample)\n",
    "    test_labels.append(label)\n",
    "\n",
    "pred_probs= make_predictions(model=model,\n",
    "                             data=test_samples)\n",
    "# Turn the prediction probabilities into prediction labels by taking the argmax()\n",
    "pred_classes = pred_probs.argmax(dim=1)\n",
    "\n",
    "# Make Predictions\n",
    "plot_predictions(test_samples, test_labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 534
    },
    "id": "oKD-gZbbld6G",
    "outputId": "1460e5f1-de03-4874-e24f-e2c9e9465f5e"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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cuJweTm97wvl++81ac94UplRqVKBE4+c0KuX9iDXrKk5V8bhKkPtp6OQN2MPXXCBdql366aNIi1NN7yRI/WBARE965s8WX/WrdMHCr7/Pric5cv1UvqWLe55HT6sG9Pq1fr3li369/r10S1D7956mei66/RAtHChyK236hPV/tsxQqlfX8vp21cfQti0SR8gfOQRfbgiN3XqJDJvnv6sWaNPKmd+EjU5WR+I2LIl+KnxUMJZTw88oNvCxo26fgYM0N+ZL/PFyimn6FBGAwbopcrLLgvEwtleC5vcbKPjx+tDJxs36rb3xBMir7wSXpJlqVxZe1/9D27t2RNdeffcI/L55zqixurV2pvy3nva7kVoz1l5+WXdhnLqlpQnn9SepTVrdJt78019Qr9s2fDb6RVX6NB8N92kJxAZk51bbtHk+5prdDk3bhT53/9E+vfPveQjv8nNfcGjj2q5kyfr9rl9u/5kHH0qEq59gc+nDxdPnx64mnXGGZorfPJJ8BWum28W+eUXbe9r1oi8+64u55135u4wjklJenXk4Yd1RJdPP9XvLKM6dXTZ5s4V+f337H1/994bfOKT2YYNesXoq690W/jgA324uUWLwO0rsda7t97+8dBDetKc8ZadUPvpKORNwuzz6Zfcvr3uiOrX17P8tLRAstmjh8i//60Lf+aZOgTSTTcFl3PwoN4vbF2O2LpVV2SLFnpGNmGC/j1wYHTLULy47jz++ENvHbjqKn1q+Nlns1fOlCm6cQ4bpvflde2qT69mfjo01vr31+S/b189qz7llODeKBFtJGlpejkk4z1uoYSznnbv1mGLGjXSexK3bNEdQaj7HnNS79769PMVVwSPKRnO9lrY5GYbPXBAD0ZNmuhJy1tv6QEs2jYaH6/3v0+apJeeM54EReKMM/RS7U8/aQ9yixY6ekbGXhXac8C6dfp0+oAB2a6+U8mSegLSqpXud/0H7Li48Ntp6dJ628XXXweerPerXl2vFB47prdlNW2qyWCZMnk/tnVeyc19wfPPa7J61VXarvw/EyZEtwzWvqBjR13f/uTY5wtcITrvvMDnatTQ72H5cu2NHjxYt+3MyWpumDxZv8dWrXT7zDxSR40a2gk1YoSuo+yMELFtm46M41K0qJ5MXHyx7uNuu02P2QsWnHgrWqycdpq2/2++ObENh7OfjpDP86K92RS5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0EH/zsn27+dlx3XXed5llwVeL1vmeRUqeF737vp61CjPS0jwvB07Ap/55BPPK13a8/76K7isevU8b9Ik/btNG88bPDg4fvbZntesWfC8GjTwvF9/tev49NNah/h4zxPxvJtuCn/5XF5+Weednh547++/PS8pyfPmz9fXGb+bvXu1Dm++Gfj87t2eV7y45w0dGnivTh3Pu/bawOv0dM+rXNnzXnhBX2/apMuwalXoOnbo4HlTpoS3PHXqeN6TTwZeX3ed51WposuUkYjnvf128HtlygTm46rfddfpPI4eDbx39dWe16NH4PUzz3hep07h1Tc730M4evXyvAsvDH7vrrs8r3HjwOtQ6yacbSK/KwjtefduzytRQttzsWL6vUcrmu22Th3P69YteLo+fTxv0KDg9xYv9ry4OM87dMjz1q/3PJ/P87ZsCf7M+ed73r33uus5apTWKRyZ1+WUKbqMq1cHf65Dh+B9kOfpdBnnk3n/kLG89esD7z33nO43/MJZp5995nk1anje779nXe9oNGum35nff//reeXKed7+/YH35s3T9bJ9u74O9/vIvM4Lmvze1hMSPG/GjOD3ZszwvKJFQy9buD7/3PMSEz3vjTf09f79+nrp0uDPDRjgeddco38vXKjb/TvvBOL7959Y38OHPa96dc979FH3/KdM0e0tHKNGBX+HWdXD87JuP0OHBs8nq23cX96CBYH35s3T9w4d0te//qrrbdmy8Orsebr85ct73m+/hT+NS1b5x513et4ll+jfEyd63lVXeV7Lllp3z/O8+vUDx+hw2n82Zb+Hee5c7UlLTNQzl/btRZ55JhCvUyf4ftgVK7QLvEIFnc7/s2mT9uSI6NlSmzbB88n8unVr7d2rUcNdt9RUPdN9/nk965szR+s7Zox7mox1ynifb0YrVujZd6lSgc+WL6+XnvzLkNHGjXrm3rp14L0yZfRhxMwyniX6fHr5ZccOd31j5fTTc/a+vCZN9PKJX7Vqwcs1ZEjOP5izeHHw+pwxI+vP/fij3rqTUdu22tt37FjgPWvdZHebyK/yc3sW0e939WrtIXjoIb1yY/V4hNOeLaG2W5FA75TfihXaC51x3hdfrL0tmzbpvsjz9FanjJ9ZtCi220rRouH1QoWreHG9NcYv83cTap3u26e9dy++qA/yhivSdfrjj3qJvUSJwHtt2+p6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",
      "text/plain": [
       "<Figure size 900x900 with 9 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "wrong_samples = []\n",
    "wrong_labels = []\n",
    "counter = 9\n",
    "\n",
    "while counter > 0:\n",
    "  randomNum = random.randint(0,len(test_data))\n",
    "  sample, label = test_data[randomNum]\n",
    "  model.eval()\n",
    "  with torch.inference_mode():\n",
    "    # Prepare sample\n",
    "    sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device\n",
    "\n",
    "    # Forward pass (model outputs raw logit)\n",
    "    pred_logit = model(sample)\n",
    "\n",
    "    # Get prediction probability (logit -> prediction probability)\n",
    "    pred_prob = torch.softmax(pred_logit.squeeze(), dim=0).cpu()\n",
    "    classNumber = torch.argmax(pred_prob).item()\n",
    "\n",
    "  if str(classNumber) != str(label):\n",
    "    wrong_samples.append(sample.cpu())\n",
    "    wrong_labels.append(label)\n",
    "    counter -= 1\n",
    "\n",
    "# Make Predictions\n",
    "plot_predictions(wrong_samples, wrong_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "m4mIR6g5KIf-"
   },
   "source": [
    "### Making Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 49,
     "referenced_widgets": [
      "b8159cc557284f2abc6fb1954e90d89a",
      "d31bb300ba6b40ff853377b5f1df6006",
      "fc581bae348147a4ba78409f5dc343dc",
      "08a78b88ee0d461fb6489c48b274c953",
      "bb11b03a836d430cb1ce3d44e6733298",
      "8b453790322b459f923d70461df2a418",
      "c834eb703b1346eaa8a29c51561573cd",
      "9b6664897b88488aba26f6cead274779",
      "70e15a1e05c84813a1b699e2e39f0925",
      "1156d5c1138343ddb8c2d983ed9b5ad7",
      "9509ba3330574bada4c9daa3479edecd"
     ]
    },
    "id": "fMqaJbxpKNU2",
    "outputId": "65d3df9f-2d9e-4bf8-dc91-f98a3b5c7a8f"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "083e667f9a6d4ec4a21467957b339c39",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Making predictions:   0%|          | 0/313 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Import tqdm for progress bar\n",
    "from tqdm.auto import tqdm\n",
    "\n",
    "# 1. Make predictions with trained model\n",
    "y_preds = []\n",
    "model.eval()\n",
    "with torch.inference_mode():\n",
    "  for X, y in tqdm(test_dataloader, desc=\"Making predictions\"):\n",
    "    # Send data and targets to target device\n",
    "    X, y = X.to(device), y.to(device)\n",
    "    # Do the forward pass\n",
    "    y_logit = model(X)\n",
    "    # Turn predictions from logits -> prediction probabilities -> predictions labels\n",
    "    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)\n",
    "    # Put predictions on CPU for evaluation\n",
    "    y_preds.append(y_pred.cpu())\n",
    "# Concatenate list of predictions into a tensor\n",
    "y_pred_tensor = torch.cat(y_preds)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "iCgIb_ozKavX",
    "outputId": "18baedee-305d-4c1a-bb62-76c9467b8484"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "mlxtend version: 0.23.3\n"
     ]
    }
   ],
   "source": [
    "# See if torchmetrics exists, if not, install it\n",
    "try:\n",
    "    import torchmetrics, mlxtend\n",
    "    print(f\"mlxtend version: {mlxtend.__version__}\")\n",
    "    assert int(mlxtend.__version__.split(\".\")[1]) >= 19, \"mlxtend verison should be 0.19.0 or higher\"\n",
    "except:\n",
    "    !pip install -q torchmetrics -U mlxtend # <- Note: If you're using Google Colab, this may require restarting the runtime\n",
    "    import torchmetrics, mlxtend\n",
    "    print(f\"mlxtend version: {mlxtend.__version__}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "lIjMywAVKfVr",
    "outputId": "0b354ec5-a322-4806-e1c9-e08e9460fb78"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.23.3\n"
     ]
    }
   ],
   "source": [
    "# Import mlxtend upgraded version\n",
    "import mlxtend\n",
    "print(mlxtend.__version__)\n",
    "assert int(mlxtend.__version__.split(\".\")[1]) >= 19 # should be version 0.19.0 or higher"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 472
    },
    "id": "aEboEBQDKhFC",
    "outputId": "b3ecc299-c77e-4086-9c36-8ed40b7c9af6"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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ZLA310ObGlnvyZUl7RWe5/PjEZy/9fkrQf8mFfD8dQpOh6nGteGAkrEdhoZcQ8vmTeT3J/69HXbv2Rur121EbGwsJo4bg/j4OFhZWaNbj14YP2mKWHELpCS8XzlJSkrEtCmT8OzZUxibmMDPrwOmzZxTJJeAU4b1a1YBgMLxEABWrlmPHr36AADatGuPpb+uwOKF8zFu9AiUr1ARf2zfDe9idjz8t+sh1+ArcJyfPGUajhw+CADwrl5N4XnHTp5F/QYNiywnUMyus9i/f3+cOXMGMTExMDQ0hJubG8aPH49mzZoVeNmzZs3CihUrEBcXh969e2PTpk05XmfRz88PRkZG2LRpE4Csr6X37dsHPz8/heWdOHECM2fORGhoKDQ0NFCpUiUMGDAAAwcOBJB11fzJkydjz549ePHiBaysrFC/fn3MmzcvV+ciFvV1FqngCvPaYmLK73UWi7vCvs6iGArzOotiKorrLIqhsK+zKJaSeiwszOssiiW311ksVmWRPmNZVD0l9QDJsqg6WBZVC8uiavkvl8WSuQcSERERkVKwLBIRERGRIJZFIiIiIhLEskhEREREglgWiYiIiEgQyyIRERERCWJZJCIiIiJBLItEREREJIhlkYiIiIgEsSwSERERkSCWRSIiIiISxLJIRERERIJYFomIiIhIEMsiEREREQliWSQiIiIiQSyLRERERCSIZZGIiIiIBLEsEhEREZEglkUiIiIiEsSySERERESC1MUOQFRSSMQOUEgSdvUXO0KhsO67VewIShezsYfYEQqFTCYTO0KhUFMrmUeNzBL6fr1PyxA7gtJ9yOU6cWSRiIiIiASxLBIRERGRIJZFIiIiIhLEskhEREREglgWiYiIiEgQyyIRERERCWJZJCIiIiJBLItEREREJIhlkYiIiIgEsSwSERERkSCWRSIiIiISxLJIRERERIJYFomIiIhIEMsiEREREQliWSQiIiIiQSyLRERERCSIZZGIiIiIBLEsEhEREZEglkUiIiIiEsSySERERESCWBaJiIiISBDL4j9cuHABrq6u0NDQgJ+fn9hxCl3w+SB09GsDJ3sb6GhIcPDAfrEjKUVJWK/g80H4pn1blHUsDV0tNRz61zoc2L8XbVu1gL2NOXS11BAeHiZKzoJas2olqldzg4WJASxMDNCgrjdOHD8mdqwvkqpJMPmbqghb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",
      "text/plain": [
       "<Figure size 1000x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from torchmetrics import ConfusionMatrix\n",
    "from mlxtend.plotting import plot_confusion_matrix\n",
    "\n",
    "# 2. Setup confusion matrix instance and compare predictions to targets\n",
    "confmat = ConfusionMatrix(num_classes=len(train_data.classes), task='multiclass')\n",
    "confmat_tensor = confmat(preds=y_pred_tensor,\n",
    "                         target=test_data.targets)\n",
    "\n",
    "# 3. Plot the confusion matrix\n",
    "fig, ax = plot_confusion_matrix(\n",
    "    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy\n",
    "    class_names=train_data.classes, # turn the row and column labels into class names\n",
    "    figsize=(10, 7)\n",
    ");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "provenance": []
  },
  "gpuClass": "standard",
  "kernelspec": {
   "display_name": "Python (pytorch_env)",
   "language": "python",
   "name": "pytorch_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.15"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "08a78b88ee0d461fb6489c48b274c953": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_1156d5c1138343ddb8c2d983ed9b5ad7",
      "placeholder": "​",
      "style": "IPY_MODEL_9509ba3330574bada4c9daa3479edecd",
      "value": " 313/313 [00:01&lt;00:00, 217.16it/s]"
     }
    },
    "0ae23734c1ac422da72ee97d8cd83bac": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_1e9cf9f7cafa4ee2a7fc2a2432ae760f",
      "max": 28881,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_5cce2e0704c841169bb468498ca20d01",
      "value": 28881
     }
    },
    "1156d5c1138343ddb8c2d983ed9b5ad7": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "1e9cf9f7cafa4ee2a7fc2a2432ae760f": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "2235cd55531342069c8e34d741c4a286": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "283196f7580c45dab33f21dba2ccb7af": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_fe6ef9b4b05d418c8e114d8e914df2bf",
      "max": 4542,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_7a5aee224a934477ae58ad93c07ac0f3",
      "value": 4542
     }
    },
    "2a1df6551f214daab9833625e2ac000f": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "3c83eca20c3046618e01064a4f81d55c": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "3f493d8dcddc48f98e44bc939980ddf7": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "4213e120a9c848d7b8b538a9b8a70f2f": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_7d934be4250f40ce9777f41685a0ae1e",
       "IPY_MODEL_6f9439de138b41e98f0781475141c582",
       "IPY_MODEL_99298dcffb0a40bfaba5584f92419b32"
      ],
      "layout": "IPY_MODEL_2235cd55531342069c8e34d741c4a286"
     }
    },
    "46f96b036f574280a5345a4b9bbb2fb2": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "5152cdbdd4c945c3b2ebe6f090f82a02": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_da9baef008544f30a7d9a6923dbacfcc",
      "placeholder": "​",
      "style": "IPY_MODEL_d5dfab5251dd46f6b7e7ef53fbbe2a38",
      "value": "100%"
     }
    },
    "545a19ff5ea8499597f894894cd20c3f": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "581655a7a9a5426ba0caaf33648b3c09": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "5cce2e0704c841169bb468498ca20d01": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "657de865fff34eceac15774ba927fbcf": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_c1ab528b7de847f98d36bce1f22c83b9",
      "placeholder": "​",
      "style": "IPY_MODEL_b70871af021b4b77aaca52471e89dd9e",
      "value": "100%"
     }
    },
    "66e5bf5e46f940b0b85aa7ceb23427ca": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "6f9439de138b41e98f0781475141c582": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_e511e17e4ece402a81736e838d5fd539",
      "max": 8,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_bc5c4704a74842579ff87d9b05e42bbd",
      "value": 8
     }
    },
    "70e15a1e05c84813a1b699e2e39f0925": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "71f70fe1622b45cebf237bb4d288a909": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_c441e9e5a6034931a17df38e0593bcc9",
      "max": 1648877,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_7abafbe26af340ac93519a610915432f",
      "value": 1648877
     }
    },
    "727ef457f40a4f07bf560fb87a67d791": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "776154d7e2484006a8aec47636d341cc": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "7917850286ad4d89bc4197248407a610": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "79da759132e84dfe9a821524e63e0989": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "7a5aee224a934477ae58ad93c07ac0f3": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "7abafbe26af340ac93519a610915432f": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "7d934be4250f40ce9777f41685a0ae1e": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_46f96b036f574280a5345a4b9bbb2fb2",
      "placeholder": "​",
      "style": "IPY_MODEL_79da759132e84dfe9a821524e63e0989",
      "value": "100%"
     }
    },
    "84d3b59879694092838841f4d32386b1": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_ffd326cbadbc4994ae8ef210b350f4b1",
      "placeholder": "​",
      "style": "IPY_MODEL_581655a7a9a5426ba0caaf33648b3c09",
      "value": "100%"
     }
    },
    "87084b3a21174948ae6b46d0b385675a": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_befc5d94a31646e6a84a4da5dc14d49b",
      "max": 9912422,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_9820c245020e4b85914ee9f4eb043ba0",
      "value": 9912422
     }
    },
    "88bb3f39ea85452a8c23cc63c9b7d589": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "8a2a44438bcc4e0f9d0fcb54804f37fb": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "8b453790322b459f923d70461df2a418": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "910702cbd449466f9a0ef7271b0e5f94": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "91334d71d68842f8ac8ac4fd030b1e6e": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_3f493d8dcddc48f98e44bc939980ddf7",
      "placeholder": "​",
      "style": "IPY_MODEL_b42ddf2713364a11bba653b17269ea2d",
      "value": "100%"
     }
    },
    "9509ba3330574bada4c9daa3479edecd": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "9820c245020e4b85914ee9f4eb043ba0": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "99298dcffb0a40bfaba5584f92419b32": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_727ef457f40a4f07bf560fb87a67d791",
      "placeholder": "​",
      "style": "IPY_MODEL_776154d7e2484006a8aec47636d341cc",
      "value": " 8/8 [01:10&lt;00:00,  8.35s/it]"
     }
    },
    "9b6664897b88488aba26f6cead274779": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "aa17afe7fb0c40a2b8aee951a002cbc3": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_5152cdbdd4c945c3b2ebe6f090f82a02",
       "IPY_MODEL_283196f7580c45dab33f21dba2ccb7af",
       "IPY_MODEL_f0fd168fa1084ab2a372e8aa8e41d8e7"
      ],
      "layout": "IPY_MODEL_3c83eca20c3046618e01064a4f81d55c"
     }
    },
    "ad4844c2efd54a9585bb76e553fa53a5": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "b283a887a32c49f68290be6616025d3a": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "b343c3207d4940acb8882ac5e587e042": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_84d3b59879694092838841f4d32386b1",
       "IPY_MODEL_87084b3a21174948ae6b46d0b385675a",
       "IPY_MODEL_caecbc38daf242a28da56061907c3832"
      ],
      "layout": "IPY_MODEL_8a2a44438bcc4e0f9d0fcb54804f37fb"
     }
    },
    "b42ddf2713364a11bba653b17269ea2d": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "b70871af021b4b77aaca52471e89dd9e": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "b8159cc557284f2abc6fb1954e90d89a": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_d31bb300ba6b40ff853377b5f1df6006",
       "IPY_MODEL_fc581bae348147a4ba78409f5dc343dc",
       "IPY_MODEL_08a78b88ee0d461fb6489c48b274c953"
      ],
      "layout": "IPY_MODEL_bb11b03a836d430cb1ce3d44e6733298"
     }
    },
    "bb11b03a836d430cb1ce3d44e6733298": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "bc5c4704a74842579ff87d9b05e42bbd": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "ProgressStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "ProgressStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "bar_color": null,
      "description_width": ""
     }
    },
    "befc5d94a31646e6a84a4da5dc14d49b": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "c1ab528b7de847f98d36bce1f22c83b9": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "c441e9e5a6034931a17df38e0593bcc9": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "c834eb703b1346eaa8a29c51561573cd": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "caecbc38daf242a28da56061907c3832": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_88bb3f39ea85452a8c23cc63c9b7d589",
      "placeholder": "​",
      "style": "IPY_MODEL_b283a887a32c49f68290be6616025d3a",
      "value": " 9912422/9912422 [00:00&lt;00:00, 7487767.74it/s]"
     }
    },
    "d31bb300ba6b40ff853377b5f1df6006": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_8b453790322b459f923d70461df2a418",
      "placeholder": "​",
      "style": "IPY_MODEL_c834eb703b1346eaa8a29c51561573cd",
      "value": "Making predictions: 100%"
     }
    },
    "d5dfab5251dd46f6b7e7ef53fbbe2a38": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "DescriptionStyleModel",
     "state": {
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "DescriptionStyleModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "StyleView",
      "description_width": ""
     }
    },
    "d6d722f7b1554b40bef8441821653e8f": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_edbf92155a6644d89aa4c11e2ea156d9",
      "placeholder": "​",
      "style": "IPY_MODEL_66e5bf5e46f940b0b85aa7ceb23427ca",
      "value": " 1648877/1648877 [00:00&lt;00:00, 5202289.21it/s]"
     }
    },
    "d6d948bc97094d9fb135ded270fabfa4": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_657de865fff34eceac15774ba927fbcf",
       "IPY_MODEL_71f70fe1622b45cebf237bb4d288a909",
       "IPY_MODEL_d6d722f7b1554b40bef8441821653e8f"
      ],
      "layout": "IPY_MODEL_545a19ff5ea8499597f894894cd20c3f"
     }
    },
    "da9baef008544f30a7d9a6923dbacfcc": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "e3d4a4cb76fc42a4b3747a82fab1c02c": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "e511e17e4ece402a81736e838d5fd539": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "e78b546c4fcd4c6d9868a734dd09e495": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HBoxModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HBoxModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HBoxView",
      "box_style": "",
      "children": [
       "IPY_MODEL_91334d71d68842f8ac8ac4fd030b1e6e",
       "IPY_MODEL_0ae23734c1ac422da72ee97d8cd83bac",
       "IPY_MODEL_ec84b44f4ecd4788b73da8041b7528cb"
      ],
      "layout": "IPY_MODEL_910702cbd449466f9a0ef7271b0e5f94"
     }
    },
    "ec84b44f4ecd4788b73da8041b7528cb": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_e3d4a4cb76fc42a4b3747a82fab1c02c",
      "placeholder": "​",
      "style": "IPY_MODEL_ad4844c2efd54a9585bb76e553fa53a5",
      "value": " 28881/28881 [00:00&lt;00:00, 630006.16it/s]"
     }
    },
    "edbf92155a6644d89aa4c11e2ea156d9": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "f0fd168fa1084ab2a372e8aa8e41d8e7": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "HTMLModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "HTMLModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "HTMLView",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_7917850286ad4d89bc4197248407a610",
      "placeholder": "​",
      "style": "IPY_MODEL_2a1df6551f214daab9833625e2ac000f",
      "value": " 4542/4542 [00:00&lt;00:00, 82361.79it/s]"
     }
    },
    "fc581bae348147a4ba78409f5dc343dc": {
     "model_module": "@jupyter-widgets/controls",
     "model_module_version": "1.5.0",
     "model_name": "FloatProgressModel",
     "state": {
      "_dom_classes": [],
      "_model_module": "@jupyter-widgets/controls",
      "_model_module_version": "1.5.0",
      "_model_name": "FloatProgressModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/controls",
      "_view_module_version": "1.5.0",
      "_view_name": "ProgressView",
      "bar_style": "success",
      "description": "",
      "description_tooltip": null,
      "layout": "IPY_MODEL_9b6664897b88488aba26f6cead274779",
      "max": 313,
      "min": 0,
      "orientation": "horizontal",
      "style": "IPY_MODEL_70e15a1e05c84813a1b699e2e39f0925",
      "value": 313
     }
    },
    "fe6ef9b4b05d418c8e114d8e914df2bf": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    },
    "ffd326cbadbc4994ae8ef210b350f4b1": {
     "model_module": "@jupyter-widgets/base",
     "model_module_version": "1.2.0",
     "model_name": "LayoutModel",
     "state": {
      "_model_module": "@jupyter-widgets/base",
      "_model_module_version": "1.2.0",
      "_model_name": "LayoutModel",
      "_view_count": null,
      "_view_module": "@jupyter-widgets/base",
      "_view_module_version": "1.2.0",
      "_view_name": "LayoutView",
      "align_content": null,
      "align_items": null,
      "align_self": null,
      "border": null,
      "bottom": null,
      "display": null,
      "flex": null,
      "flex_flow": null,
      "grid_area": null,
      "grid_auto_columns": null,
      "grid_auto_flow": null,
      "grid_auto_rows": null,
      "grid_column": null,
      "grid_gap": null,
      "grid_row": null,
      "grid_template_areas": null,
      "grid_template_columns": null,
      "grid_template_rows": null,
      "height": null,
      "justify_content": null,
      "justify_items": null,
      "left": null,
      "margin": null,
      "max_height": null,
      "max_width": null,
      "min_height": null,
      "min_width": null,
      "object_fit": null,
      "object_position": null,
      "order": null,
      "overflow": null,
      "overflow_x": null,
      "overflow_y": null,
      "padding": null,
      "right": null,
      "top": null,
      "visibility": null,
      "width": null
     }
    }
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
