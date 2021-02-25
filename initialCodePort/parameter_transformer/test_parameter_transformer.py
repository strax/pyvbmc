import pytest
import numpy as np
from parameter_transformer import ParameterTransformer

"""
    toDo:
    - test init < < for bounds
"""

"""
__Init__ method
"""
NVARS = 3


def test_init_no_lower_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(np.isinf(parameter_transformer.lower_bounds_orig))


def test_init_lower_bounds():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS, lower_bounds=np.ones((1, NVARS))
    )
    assert np.all(parameter_transformer.lower_bounds_orig == np.ones(NVARS))


def test_init_no_upper_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(np.isinf(parameter_transformer.upper_bounds_orig))


def test_init_upper_bounds():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS, upper_bounds=np.ones((1, NVARS))
    )
    assert np.all(parameter_transformer.upper_bounds_orig == np.ones(NVARS))


def test_init_type_1():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -np.inf,
        upper_bounds=np.ones((1, NVARS)) * np.inf,
    )
    assert np.all(parameter_transformer.type == np.zeros(NVARS))


def test_init_type_1():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)),
        upper_bounds=np.ones((1, NVARS)) * np.inf,
    )
    assert np.all(parameter_transformer.type == np.ones(NVARS))


def test_init_type_2():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -np.inf,
        upper_bounds=np.ones((1, NVARS)),
    )
    assert np.all(parameter_transformer.type == np.ones(NVARS) * 2)


def test_init_type_3():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)),
        upper_bounds=np.ones((1, NVARS)) * 2,
    )
    assert np.all(parameter_transformer.type == np.ones(NVARS) * 3)


def test_init_mu_inf_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(parameter_transformer.mu == np.zeros(NVARS))


def test_init_delta_inf_bounds():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    assert np.all(parameter_transformer.delta == np.ones(NVARS))


def test_direct_transform_type3_within():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    X = np.ones((10, NVARS)) * 3
    Y = parameter_transformer.direct_transform(X)
    Y2 = np.ones((10, NVARS)) * 0.619
    assert np.all(np.isclose(Y, Y2, atol=1e-04))


def test_direct_transform_type3_within_negative():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -10,
        upper_bounds=np.ones((1, NVARS)) * 10,
    )
    X = np.ones((10, NVARS)) * -4
    Y = parameter_transformer.direct_transform(X)
    Y2 = np.ones((10, NVARS)) * -0.8473
    assert np.all(np.isclose(Y, Y2))


def test_direct_transform_type0():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    X = np.ones((10, NVARS)) * 3
    Y = parameter_transformer.direct_transform(X)
    assert np.all(Y == X)


def test_direct_transform_type0_negative():
    parameter_transformer = ParameterTransformer(nvars=NVARS)
    X = np.ones((10, NVARS)) * -4
    Y = parameter_transformer.direct_transform(X)
    assert np.all(Y == X)

def test_direct_transform_type1():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)),
        upper_bounds=np.ones((1, NVARS)) * np.inf,
    )
    X = np.ones((10, NVARS)) * 3
    Y = parameter_transformer.direct_transform(X)
    assert np.all(Y == X)


def test_direct_transform_type1_negative():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)),
        upper_bounds=np.ones((1, NVARS)) * np.inf,
    )
    X = np.ones((10, NVARS)) * -4
    Y = parameter_transformer.direct_transform(X)
    assert np.all(Y == X)

def test_direct_transform_type2():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -np.inf,
        upper_bounds=np.ones((1, NVARS)),
    )
    X = np.ones((10, NVARS)) * 3
    Y = parameter_transformer.direct_transform(X)
    assert np.all(Y == X)


def test_direct_transform_type2_negative():
    parameter_transformer = ParameterTransformer(
        nvars=NVARS,
        lower_bounds=np.ones((1, NVARS)) * -np.inf,
        upper_bounds=np.ones((1, NVARS)),
    )
    X = np.ones((10, NVARS)) * -4
    Y = parameter_transformer.direct_transform(X)
    assert np.all(Y == X)

