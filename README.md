# get trips from validations

This script reads validation from TMSA (Google BigQuery) and obtain the trips based on the following definition:
Trip: Sum of the value (cop) of the first validation (full fare) and the following validations (transshipment value) within 90 minutes after the first validation.

Execute:

python run.py
