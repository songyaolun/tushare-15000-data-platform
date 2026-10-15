"""P02 七因子、事件解析、样本收缩和严格主榜测试。"""

import pytest

from tushare_entitlement_hub.cross_asset_research.theme_ranking import (
    ThemeObservation,
    average_tie_percentiles,
    calculate_theme_features,
    normalize_reason_tokens,
    parse_board_count,
    rank_theme_features,
)


def observation(code="T1", members=20, flow=10.0, heat=True):
    """构造完整主题观察；输入成员数，输出可进入严格主榜的确定性样本。"""
    member_codes = tuple(f"S{index}" for index in range(members))
    return ThemeObservation(
        theme_code=code,
        theme_name=code,
        member_codes=member_codes,
        returns_by_member={member: 0.01 for member in member_codes},
        flow_strength=flow,
        heat_by_member={member: 1 / (index + 1) for index, member in enumerate(member_codes)}
        if heat
        else {},
        limit_known_members=frozenset(member_codes),
        limit_up_members=frozenset(member_codes[:2]),
        board_counts={member_codes[0]: 3, member_codes[1]: 1},
        reason_tokens_by_member={member_codes[0]: ("ai",), member_codes[1]: ("ai",)},
        evidence_ids=("e1", "e1", "e2"),
    )


@pytest.mark.parametrize(
    "status,expected",
    [("首板", 1), ("3连板", 3), ("5天3板", 3), (" 2 板 ", 2), (None, 0), ("异动", 0)],
)
def test_board_status_is_parsed_without_free_form_guessing(status, expected):
    """场景：首板、连板、天数板和无关文本；预期：只解析确定格式。"""
    assert parse_board_count(status) == expected


def test_reason_normalization_prefers_theme_and_falls_back_to_description():
    """场景：结构化主题、多分隔符及说明回退；预期：小写、去符号、去重排序。"""
    assert normalize_reason_tokens("AI+传媒，国企+AI", "忽略") == ("ai", "传媒", "国企")
    assert normalize_reason_tokens(None, " 回购 预期 ") == ("回购预期",)
    assert normalize_reason_tokens(None, None) == ()


def test_features_apply_coverage_gates_and_event_formulas():
    """场景：20名成员、两只涨停且一只三板；预期：七因子和证据ID准确。"""
    features = calculate_theme_features(observation())
    assert features["main_eligible"] and features["price_coverage"] == 1
    assert features["limit_up_breadth"] == 0.1
    assert features["board_strength"] == 0.025
    assert features["reason_consistency"] == 1
    assert features["evidence_ids"] == ["e1", "e2"]
    incomplete = observation(members=5)
    incomplete = ThemeObservation(
        **{**incomplete.__dict__, "returns_by_member": {"S0": 0.01, "S1": 0.01}}
    )
    degraded = calculate_theme_features(incomplete)
    assert not degraded["price_eligible"]
    assert "effective_members_below_five" in degraded["reason_codes"]
    limit_incomplete = observation()
    limit_incomplete = ThemeObservation(
        **{
            **limit_incomplete.__dict__,
            "limit_known_members": frozenset(limit_incomplete.member_codes[:10]),
        }
    )
    assert (
        "limit_coverage_below_90_percent"
        in calculate_theme_features(limit_incomplete)["reason_codes"]
    )


def test_ranking_requires_global_readiness_and_shrinks_small_themes():
    """场景：5名强主题与20名弱主题；预期：主榜受全局门禁控制，小样本得分向50收缩。"""
    small = calculate_theme_features(observation("small", members=5, flow=20))
    large = calculate_theme_features(observation("large", members=20, flow=10))
    main, price = rank_theme_features([small, large], release_ready=False)
    assert main == [] and len(price) == 2
    main, _ = rank_theme_features([small, large], release_ready=True)
    assert len(main) == 2
    small_row = next(item for item in main if item["theme_code"] == "small")
    assert small_row["sample_shrinkage"] == 0.25
    assert small_row["score"] == pytest.approx(50 + (small_row["raw_score"] - 50) * 0.25)
    assert {item["rank"] for item in main} == {1, 2}


def test_average_tie_percentiles_handles_ties_singletons_and_bad_numbers():
    """场景：并列、单样本和非有限值；预期：平均秩、0.5及明确拒绝。"""
    assert average_tie_percentiles({"a": 1, "b": 2, "c": 2}) == {
        "a": 0,
        "b": 0.75,
        "c": 0.75,
    }
    assert average_tie_percentiles({"a": 1}) == {"a": 0.5}
    with pytest.raises(ValueError, match="non_finite_percentile_input"):
        average_tie_percentiles({"a": float("nan")})
