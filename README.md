# Climate ML Pipeline

Projeto de estudo para desenvolvimento de um pipeline completo de dados e Machine Learning, utilizando dados meteorológicos obtidos através de uma API.

## Objetivo

Construir, de forma incremental, um pipeline capaz de:

* Coletar dados de uma API meteorológica;
* Realizar limpeza e validação dos dados;
* Armazenar os dados em PostgreSQL;
* Realizar Feature Engineering;
* Treinar e avaliar modelos de Machine Learning;
* Automatizar o pipeline com Airflow;
* Containerizar a aplicação com Docker;
* Disponibilizar previsões através de uma API com FastAPI.

## Pipeline

```text
API Meteorológica
       ↓
Python + Requests
       ↓
Pandas
       ↓
Limpeza e Validação
       ↓
PostgreSQL
       ↓
Feature Engineering
       ↓
Scikit-Learn
       ↓
Modelo de ML
       ↓
FastAPI
```

## Tecnologias

* Python
* Requests
* Pandas
* PostgreSQL
* SQLAlchemy
* Scikit-Learn
* Apache Airflow
* Docker
* FastAPI

## Status

🚧 Em desenvolvimento.

O projeto será desenvolvido de forma incremental, adicionando novas etapas e tecnologias conforme o pipeline evolui.
