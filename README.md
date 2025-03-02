# Shoeprint Image Retrieval and Crime Scene Shoeprint Image Linking

This project uses Convolutional Neural Networks (CNN) and normalized cross-correlation along with ViTs to retrieve and link shoeprint images from crime scenes.


## Setup

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Ensure you have the necessary dataset from the links given below.

## Usage

### Creating the Dataset

Run the `create-dataset.py` to turn the images present at dataset https://iastate.figshare.com/articles/figure/2D_Footwear_outsole_impressions/11624073/1?file=21217842)

```sh
python create-dataset.py
```

### Training the Model
There are mainly two models that are being trained using the code in this repository: `msn.ipynb` and `vit-test.ipynb`. Both of the **Vision Transfomers** are trained on this [dataset](https://fid.dmi.unibas.ch/).


### Testing the Model

Use the `test.ipynb` notebook to test the model with new images. Change the variable `pretrained_model_name` to the desired model.

### Using the Retreival Code

The code mentioned the directory `retrieval` contains the vector embedding implementation.
`ResNet-RetrievalTest.ipynb` contains the code for using embeddings generated using the `ResNet-50` model while `ViTRestrievalTest.ipynb` contains the code for using the embeddings generated using a Vision Transformers (un-fine-tuned).

The vector dataabase is created on the following [dataset](https://iastate.figshare.com/articles/figure/2D_Footwear_outsole_impressions/11624073/1?file=21217842)

`Data-information.xlsx` contains the mapping for the Shoe make and models.

Given below is the code for retrieving images based on ViTs

#### Example

```python
# Example code to test the model
test  = [Image.open("../patch.png").convert("RGB")]

try:
    test_inputs = processor(test, return_tensors="pt")
    test_outputs = model(**test_inputs)
except Exception as e:
    print(e)
embeddings = test_outputs.last_hidden_state[:, 0, :]  # Take CLS token embedding

# Convert to list
test_embeddings = embeddings.detach().numpy().tolist()
search_result = qclient.query_points("shoeprints_part1", query=test_embeddings[0])
return_retrived_image(search_result)
```

## License

This project is licensed under the MIT License.
