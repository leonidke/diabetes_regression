This project uses the diabetes dataset from sklearn to build a linear regression model. The goal is to predict the disease progression based on various factors.

## Setup

1. Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

To train the model and visualize the results, run:
```bash
python src/train_model.py
```

## Project Structure

```
diabetes_regression/
|-- data/
|-- notebooks/
|-- src/
|   |-- train_model.py
|-- .gitignore
|-- README.md
|-- requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

