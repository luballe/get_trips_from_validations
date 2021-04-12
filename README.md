# get trips from validations

This script reads validation from TMSA (Transmilenio - Google BigQuery) and obtain the trips based on the following definition:
Trip: Sum of the value (cop) of the first validation (full fare) and the following validations (transshipment value) within 95 minutes after the first validation (for 2019 110 mins for 2020).

To Execute:

python run.py
