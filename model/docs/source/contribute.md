# Contribute Process

## Freeze packages

- for dev, generate `requirements.txt`

```sh
# run under project root
pip freeze > requirements.txt
```

- for prod, generate `requirements.txt`

```sh
# run under project root
pip freeze > prod_requirements.txt
```

## Remove Git Branches other than main under Linux/Mac

```sh
shell/git/delete-branches.sh
```

## Run Tests

- add tests in `./src/tests` with the same relative folder structure as the source python module to test (or target python module)

- target python module cannot be placed in the project root

- test files need to be initial with `test_` to be included

- run tests on the project folder

```sh
pytest
```

## Update docker image

- see [`docs/docker.md`](./docker.md#workflow)
