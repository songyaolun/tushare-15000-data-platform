"""统一P03父结果索引的操作数与确定性回归。"""

import hashlib

from test_research_candidates import research_inputs, research_parents

from tushare_entitlement_hub.equity_research.candidates import build_candidates, index_parents


class SinglePassParents(tuple):
    """第二次遍历立即失败，检测按实体重复扫描父结果。"""

    def __new__(cls, values):
        """包装父集合并初始化计数；例：三条父结果初始visits=0。"""
        instance = super().__new__(cls, values)
        instance.passes = 0
        instance.visits = 0
        return instance

    def __iter__(self):
        """只允许一次扫描；例：第二次iter触发断言。"""
        self.passes += 1
        assert self.passes == 1
        for value in super().__iter__():
            self.visits += 1
            yield value


def test_parent_index_is_single_pass_at_production_entity_count():
    """场景：5911个父结果；预期：索引仅扫描5911次，重复身份标记未知。"""
    context = research_inputs(5911)
    parents = research_parents(context, quality=80)[0]
    group = SinglePassParents(parents)
    indexed = index_parents(group)
    assert group.visits == 5911 and group.passes == 1
    assert len(indexed) == 5911
    assert index_parents((parents[0], parents[0]))[parents[0].envelope.entity_id] is None


def test_all_three_parent_groups_are_scanned_once_and_results_repeat():
    """场景：32实体缺历史窗口；预期：三组父记录只访问96次，重复运行结果逐字一致。"""
    context = research_inputs(32)
    original = research_parents(context, quality=80)
    groups = tuple(SinglePassParents(group) for group in original)
    first = build_candidates(context, *groups)
    second = build_candidates(context, *original)
    assert sum(group.visits for group in groups) == 96
    assert [item.to_json() for item in first] == [item.to_json() for item in second]


def candidate_fingerprint(results):
    """逐条累计完整JSON摘要；例：改变任一贡献或引用会改变结果指纹，不复制结果集合。"""
    fingerprint = hashlib.sha256()
    for result in results:
        payload = result.to_json().encode("utf-8")
        fingerprint.update(len(payload).to_bytes(8, "big"))
        fingerprint.update(payload)
    return fingerprint.hexdigest()
