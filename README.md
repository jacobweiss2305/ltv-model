# Lifetime Value Model

## Overview
Can we build an accurate machine learning model to predict next 6 month LTV at customer level?

## Analysis
- Please see ./ltv_model/model/notebooks/main.ipynb
- Steps to interact with notebook:
  - git clone https://github.com/jacobweiss2305/ltv_model.git
  - cd ltv_model
  - python -m venv venv (see [provision](./model/docs/provision.md) for virtualenv installation)
  - pip install -r requirements.txt
  - source venv/bin/activate
  - If you want to run Causal Graphical Model and Bayesian Network:
    - Please see installation guide for pygraphiz: https://pygraphviz.github.io/documentation/stable/install.html
  - To spin up Jupyter lab via kedro:
       ```
       cd ./ltv_model/model/

       kedro jupyter lab
       ```
## Provision
- I recommend using pyenv for python version control. We are running 3.8.9. Please see [provision](./model/docs/provision.md) for details (model/docs/provision.md).
