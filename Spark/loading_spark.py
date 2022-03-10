import gzip
import json
import pandas as pd
import spacy

from pyspark.sql import SparkSession

from pyspark import SparkFiles

spark = SparkSession.builder.appName('nlp_spark').getOrCreate()

fileloc = '/Users/lukemoynihan/Downloads/nq-train-01.json'

df = spark.read.json(fileloc)

df.show(n=3)
