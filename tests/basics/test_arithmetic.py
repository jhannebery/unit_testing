import pytest

from src.basics.arithmetic import add, multiply, power, subtract

@pytest.fixture
def input_data():
    return [1,2]

def test_add(input_data):
    assert(add(input_data)==3)

def test_multiply(input_data):
    assert(multiply(input_data)==2)

def test_power(input_data):
    assert(power(input_data)==1)

def test_subtract(input_data):
    assert(subtract(input_data)==-1)