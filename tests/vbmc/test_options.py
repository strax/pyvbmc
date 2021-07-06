import pytest
from pyvbmc.vbmc import Options


def test_options_no_user_options():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    options = Options(default_options_path, {"D": 2})
    assert options.get("sgdstepsize") == 0.005
    assert len(options.get("useroptions")) == 0
    assert options.get("display") == "iter"


def test_options_user_options():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    user_options = {"display": "off"}
    options = Options(default_options_path, {"D": 2}, user_options)
    assert options.get("display") == "off"
    assert options.get("sgdstepsize") == 0.005
    assert len(options.get("useroptions")) == 1
    assert "display" in options.get("useroptions")


def test_init_from_existing_options():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    user_options = {"display": "off"}
    options_1 = Options(default_options_path, {"D": 2}, user_options)
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 == options_2
    assert len(options_1) == len(options_2)


def test_init_from_existing_options_modified():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    user_options = {"display": "off"}
    options_1 = Options(default_options_path, {"D": 2}, user_options)
    options_1["sgdstepsize"] = 0.3
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 != options_2
    assert options_1.get("sgdstepsize") == 0.3
    assert options_2.get("sgdstepsize") == 0.005
    assert options_1.get("display") == "off"
    assert options_2.get("display") == "off"


def test_init_from_existing_options_without_user_options():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    options_1 = Options(default_options_path, {"D": 2})
    options_1["sgdstepsize"] = 0.3
    options_2 = Options.init_from_existing_options(
        default_options_path, {"D": 2}, options_1
    )
    assert options_1 != options_2
    assert options_1.get("sgdstepsize") == 0.3
    assert options_2.get("sgdstepsize") == 0.005
    assert options_1.get("display") == "iter"
    assert options_2.get("display") == "iter"


def test_init_from_existing_options_without_other_options():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    options_1 = Options.init_from_existing_options(
        default_options_path, {"D": 2}
    )
    options_2 = Options(default_options_path, {"D": 2})
    assert options_1 == options_2
    assert len(options_1) == len(options_2)


def test_str():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    options = Options(default_options_path, {"D": 2})
    one_option_str = "funevalstart: 10 (Number of initial target fcn evals)"
    assert one_option_str in options.__str__()
    assert len(options.__str__()) > 1600


def test_del():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    user_options = {"display": "off"}
    options = Options(default_options_path, {"D": 2}, user_options)
    options.pop("display")
    assert "display" not in options


def test_eval_callable():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"

    def bar_function(T, S):
        return S, T

    user_options = {"foo": lambda Y, K: (Y, K), "bar": bar_function}
    options = Options(default_options_path, {"D": 2}, user_options)
    assert (2, 3) == options.eval("foo", {"K": 3, "Y": 2})
    assert (2, 3) == options.eval("foo", {"Y": 2, "K": 3})
    assert (3, 2) == options.eval("bar", {"T": 2, "S": 3})
    assert (5, 10) == options.eval("bar", {"S": 5, "T": 10})


def test_eval_constant():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    user_options = {"nsent": (5, 3)}
    options = Options(default_options_path, {"D": 2, "Y": 3}, user_options)
    assert (5, 3) == options.eval("nsent", {"K": 2})


def test_eval_callable_args_missing():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    user_options = {"nsent": lambda Y, K: (Y, K)}
    options = Options(default_options_path, {"D": 2}, user_options)
    with pytest.raises(TypeError):
        options.eval("nsent", {})


def test_eval_callable_too_many_args():
    default_options_path = "./pyvbmc/vbmc/option_configs/test_options.ini"
    user_options = {"bar": lambda Y, K: (Y, K)}
    options = Options(default_options_path, {"D": 2}, user_options)
    with pytest.raises(TypeError):
        options.eval("bar", {"U": 2, "S": 2, "T": 4})
