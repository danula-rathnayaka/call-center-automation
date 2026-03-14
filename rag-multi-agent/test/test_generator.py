import csv

import pytest

from multiagent_rag.agents.generator import Generator


def load_data():
    with open("data/Generator_Eval_Dataset.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def generator():
    return Generator()


@pytest.mark.parametrize("row", load_data())
def test_generator(generator, row):
    query = row["query"]
    context = row["provided_context"]

    result = generator.generate(query, context, [])

    assert isinstance(result, str)
    assert len(result.strip()) > 0
    assert "System Error" not in result
