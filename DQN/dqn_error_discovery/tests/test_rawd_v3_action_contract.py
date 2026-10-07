import numpy as np

from dqn_error_discovery.v3.action_contract import ACTION_CONTRACTS, V3_ACTION_TYPES, is_fill_input_target, is_select_option_target
from dqn_error_discovery.v3.action_space import V3ActionSpace
from dqn_error_discovery.v3.dimensions import v3_dimensions
from dqn_error_discovery.v3.select_option_executor import execute_select_option


def candidate(tag, role="", **extra):
    return {"bid": "x", "tag": tag, "role": role, "bbox": [0, 0, 100, 24], "visible": True, "enabled": True, **extra}


def native_select_candidate(**extra):
    return candidate("", "combobox", canonical_dom_resolved=True, canonical_dom_tag="select", canonical_native_control_kind="native_select", **extra)


def test_native_select_contract_is_not_fill_input():
    select = native_select_candidate()
    assert not is_fill_input_target(select)
    assert is_select_option_target(select)


def test_textarea_and_text_input_remain_fillable():
    assert is_fill_input_target(candidate("input", "textbox", input_type="text", fillable=True))
    assert is_fill_input_target(candidate("textarea", "textbox"))


def test_disabled_select_is_not_eligible():
    assert not is_select_option_target(native_select_candidate(disabled=True))


def test_custom_combobox_is_not_native_select():
    combo = candidate("div", "combobox", clickable=True)
    assert not is_select_option_target(combo)
    assert not is_fill_input_target(combo)


def test_v3_mask_and_dimensions():
    space = V3ActionSpace()
    select = native_select_candidate()
    text = candidate("input", "textbox", input_type="text", fillable=True)
    mask = space.build_action_mask({"candidate_elements": [select, text], "page_state": {"url": "http://localhost:9225/"}, "history": {"step_index": 6}})
    assert mask.shape == (640,)
    assert mask[space.encode("fill_input", 0)] == 0
    assert mask[space.encode("select_option", 0)] == 1
    assert mask[space.encode("fill_input", 1)] == 1
    assert v3_dimensions() == {"ppo_obs_dim": 635, "confirmation_obs_dim": 73, "history_v2_dim": 276, "dqn_obs_dim": 984, "action_type_count": 20, "candidate_count": 32, "action_dim": 640}


def test_every_v3_action_has_a_contract_and_executor():
    assert set(ACTION_CONTRACTS) == set(V3_ACTION_TYPES)
    assert all(contract.executor for contract in ACTION_CONTRACTS.values())


class FakeOption:
    def __init__(self, value, text, disabled=False):
        self.value, self.text, self.disabled = value, text, disabled

    def get_attribute(self, name):
        if name == "value": return self.value
        if name == "disabled" and self.disabled: return "disabled"
        return None

    def text_content(self): return self.text


class FakeOptions:
    def __init__(self): self.items = [FakeOption("", ""), FakeOption("a", "A"), FakeOption("b", "B")]
    def count(self): return len(self.items)
    def nth(self, index): return self.items[index]


class FakeSelect:
    def __init__(self): self.value = ""; self.options = FakeOptions()
    def input_value(self): return self.value
    def locator(self, selector): assert selector == "option"; return self.options
    def select_option(self, *, value): self.value = value


class FakePage:
    def __init__(self): self.select = FakeSelect()
    def locator(self, selector): assert selector == '[bid="x"]'; return self.select


def test_native_select_executor_changes_value():
    page = FakePage()
    result = execute_select_option(page, candidate("select", "combobox"))
    assert result["selected_option"]["value"] == "a"
    assert page.select.value == "a"
