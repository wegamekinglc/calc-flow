# Calc Flow

## Introduction

Calc Flow is a micro batch / streaming stateful calculation engine based on dataframe or array.

## Data Flow

Data will be ingested in one of the two formats:

1. apache arrow table or record batch;
2. a python array api compatible array object;

## Calculation Mode

It supports both micro-batch (dozens to tens of thousands) and streaming (one batch per-round) mode.

## Calculation Engine

1. dataframe engine for apache arrow table/record batch;
2. array engine for array.

## Key Requirements

* using apache arrow table or record batches as the internal dataframe in memory format;
* using standard python array api to manipulate the array;
* input data can be apache arrow table, record batches or a compatible array;
* should at least support pandas, polars and datafusion as the dataframe calculation engines;
* should at least support numpy, jax as the array calculation engines;
* should have a builtin mechanism for states recovery, e.g. checkpoints.
