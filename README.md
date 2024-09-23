  

### README 

  

This README provides instructions on how to set up and run the method extraction and N-gram model scripts for Java code completion. 

  

**Step 1: Clone the Repository** 

  

To start, clone the Spring Boot repository (or any other large Java-based repository): 

  

```

git clone https://github.com/spring-projects/spring-boot.git 

cd spring-boot 

``` 

  

**Step 2: Run Method Extraction** 

  

Use the `tokenize_methods.py` script to extract Java method signatures and bodies. This script will generate a file named `tokenized_methods.txt`, containing the tokenized method names and bodies: 

  

```

python tokenize_methods.py 

``` 

  

The script skips files larger than 10 MB and only extracts methods with at least three tokens. Methods with fewer than three tokens are skipped to ensure compatibility with Bigram and Trigram models. 

  

**Step 3: Apply N-Gram Models** 

  

Once you have the `tokenized_methods.txt` file, you can apply the N-gram model using the `ngram_model.py` script. This script allows you to choose between Unigram, Bigram, and Trigram models, or run all three models in sequence: 

  

```

python ngram_model.py 

``` 

  

When prompted, specify which model to run: 

  

``` 

Please specify the N-gram model (Unigram, Bigram, Trigram, All): All 

``` 

  

The script will output Precision, Recall, and F1 Score for each model. 

  

**Dependencies:** 

  

- Python 3.x 

- `sklearn` 

- `tqdm` (for progress monitoring) 

  

**Command for Installation:** 

  

You can install the required packages using: 

  

```bash 

pip install sklearn
pip install tqdm

``` 

 
