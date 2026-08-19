import json

import pytest

from resnet_ablation.config import Config


def _write(tmp_path, text: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return str(p)


def test_from_yaml_valid_roundtrip(tmp_path):
    cfg = Config.from_yaml(
        _write(
            tmp_path,
            "model:\n  arch: resnet20_cifar\n  attention: se\n",
        )
    )
    assert isinstance(cfg, Config)
    assert cfg.model.arch == "resnet20_cifar"
    assert cfg.model.attention == "se"
    # defaults survive for untouched groups
    assert cfg.optim.sched == "cosine"
    assert cfg.train.epochs == 200
    assert cfg.data.aug == "standard"


def test_from_yaml_missing_groups_use_defaults(tmp_path):
    cfg = Config.from_yaml(_write(tmp_path, "model:\n  arch: resnet18\n"))
    assert cfg.model.arch == "resnet18"
    assert cfg.optim.name == "sgd"
    assert cfg.data.name == "cifar10"


def test_from_yaml_unknown_group_key_raises(tmp_path):
    p = _write(tmp_path, "model:\n  arch: resnet18\ntrainings:\n  epochs: 1\n")
    with pytest.raises(ValueError, match="trainings"):
        Config.from_yaml(p)


def test_from_yaml_unknown_key_raises(tmp_path):
    p = _write(tmp_path, "model:\n  not_a_field: 1\n")
    with pytest.raises(ValueError, match="not_a_field"):
        Config.from_yaml(p)


def test_from_yaml_invalid_enum_raises(tmp_path):
    p = _write(tmp_path, "model:\n  attention: bananas\n")
    with pytest.raises(ValueError, match="attention"):
        Config.from_yaml(p)


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("model", "arch", "resnet99"),
        ("model", "shortcut", "D"),
        ("model", "stem", "banana"),
        ("model", "downsample", "bogus"),
        ("model", "attention", "bananas"),
        ("optim", "sched", "exponential"),
        ("data", "name", "coco"),
        ("data", "aug", "extreme"),
    ],
)
def test_from_yaml_invalid_enum_across_groups(tmp_path, group, key, value):
    p = _write(tmp_path, f"{group}:\n  {key}: {value}\n")
    with pytest.raises(ValueError, match=key):
        Config.from_yaml(p)


def test_config_as_dict_jsonable():
    obj = json.loads(json.dumps(Config().as_dict()))
    assert set(obj) == {"model", "optim", "train", "data"}
    assert obj["model"]["arch"] == "resnet20_cifar"
    assert obj["optim"]["sched"] == "cosine"


def test_config_as_dict_preserves_overrides(tmp_path):
    cfg = Config.from_yaml(_write(tmp_path, "model:\n  arch: resnet50\n  attention: eca\n"))
    d = cfg.as_dict()
    assert d["model"]["arch"] == "resnet50"
    assert d["model"]["attention"] == "eca"


def test_validate_rejects_bad_enum():
    cfg = Config()
    cfg.model.attention = "bananas"
    with pytest.raises(ValueError, match="attention"):
        cfg.validate()


def test_validate_accepts_good_enum(tmp_path):
    cfg = Config.from_yaml(_write(tmp_path, "model:\n  arch: resnet152\n  shortcut: C\n"))
    cfg.validate()  # must not raise