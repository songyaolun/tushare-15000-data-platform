"""公开文档发现、权益证据和离线重放测试，全程隔离真实网络。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from tushare_entitlement_hub import catalog
from tushare_entitlement_hub.models import CatalogEntry, Policy


def write_document(directory: Path, filename: str, body: str) -> Path:
    """创建独立文档输入，返回其路径；例如 doc-27.html 与 daily 正文生成可重放文件。

    输入目录由 pytest 管理，正文是公开文档的最小语义片段；不包含网络或账户凭据。
    """
    path = directory / filename
    path.write_text(body, encoding="utf-8")
    return path


def test_load_policies_and_duplicate_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """正式政策包含具体主键，重复接口名应拒绝；例如两个 daily 条目触发 ValueError。"""
    policies = catalog.load_policies()
    assert len(policies) == 250
    assert policies["daily"].primary_key == ["ts_code", "trade_date"]
    assert policies["income_vip"].minimal_params == {"period": "20231231", "ts_code": "000001.SZ"}
    assert policies["pro_bar"].lifecycle == "blocked"
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    (data_directory / "policies.json").write_text(
        '[{"api_name":"daily"},{"api_name":"daily"}]', encoding="utf-8"
    )
    monkeypatch.setattr(catalog, "__file__", str(tmp_path / "catalog.py"))
    with pytest.raises(ValueError, match="政策接口重复：daily"):
        catalog.load_policies()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/document/2?doc_id=27", "https://tushare.pro/document/2?doc_id=27"),
        (
            "https://tushare.pro:443/document/2?doc_id=27#anchor",
            "https://tushare.pro/document/2?doc_id=27",
        ),
        ("/document/1", "https://tushare.pro/document/1"),
        ("https://tushare.pro/document/2?doc_id=abc", None),
        ("https://tushare.pro:bad/document/2", None),
        ("https://[bad/document/2", None),
        ("https://tushare.pro:444/document/2", None),
        ("https://person:placeholder@tushare.pro/document/2", None),
        ("http://tushare.pro/document/2", None),
        ("https://api.tushare.pro/document/2", None),
        ("https://tushare.pro/not-document", None),
    ],
)
def test_document_url_allowlist(value: str, expected: str | None) -> None:
    """URL 仅允许无凭据的官方 HTTPS 文档；例如 API 域与非法端口均返回 None。"""
    assert catalog._document_url(value) == expected


@pytest.mark.parametrize("suffix", [".md", ".markdown"])
def test_body_markdown_and_navigation_removal(suffix: str) -> None:
    """Markdown 表格可读且导航脚本不进入正文；例如 daily 正文保留而 nav 中的 foo 被移除。"""
    body = catalog._body(
        "# 日线\n\n接口：daily\n\n| 字段 | 含义 |\n| --- | --- |\n| close | 收盘 |", suffix
    )
    assert body.find("table") is not None
    html = (
        "<nav>接口：foo</nav><article><h1>日线</h1><p>接口：daily</p>"
        "<script>bad()</script></article>"
    )
    cleaned = catalog._body(html, ".html")
    assert "foo" not in cleaned.get_text()
    assert "bad" not in cleaned.get_text()
    assert catalog._api_names(cleaned) == ["daily"]


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p>接口：daily</p><pre>pro.daily()</pre>", ["daily"]),
        ("<p>接口：income</p><p>请使用income_vip接口</p>", ["income", "income_vip"]),
        ("<p>接口：rt_min</p><pre>pro.rt_min_daily()</pre>", ["rt_min", "rt_min_daily"]),
        ("<p>输入参数</p><pre>pro.query('daily')</pre>", ["daily"]),
        ("<p>接口：行情目录</p>", []),
        ("<p>仅说明pro.daily()的介绍页面</p>", []),
        ("<p>接口：alpha</p><p>接口：beta</p>", ["alpha", "beta"]),
    ],
)
def test_api_names_preserve_declared_interfaces(html: str, expected: list[str]) -> None:
    """显式接口、VIP 与同页多声明均完整发现；例如相邻 alpha、beta 声明输出两个名称。"""
    assert catalog._api_names(catalog._body(html, ".html")) == expected


def test_numeric_claims_and_paragraph_boundaries() -> None:
    """数值只按给定语义提取，表格段落不混入正文；例如 6,000 行输出整数 6000。"""
    assert catalog._number("最多6,000行", (r"最多([\d,]+)行",)) == 6000
    assert catalog._number("价格200元", (r"每分钟(\d+)次",)) is None
    body = catalog._body(
        "<p>积分：2000积分<br>income_vip需要5000积分</p><p>积分：2000积分</p>"
        "<table><tr><td><p>忽略表格</p></td></tr></table>",
        ".html",
    )
    assert catalog._paragraph_sentences(body) == ["积分：2000积分", "income_vip需要5000积分"]
    standard_claims = catalog._claims(body, "income")
    vip_claims = catalog._claims(body, "income_vip")
    assert [claim["minimum_points"] for claim in standard_claims] == [2000]
    assert [claim["minimum_points"] for claim in vip_claims] == [5000]


def test_claims_keep_limits_frequency_tiers_and_lifecycle() -> None:
    """阶梯频次按 15000 积分选择，停更与独立权限分别保留；例如 8000 档映射每分钟 500 次。"""
    body = catalog._body(
        "<p>限量：单次最大6000</p><p>积分：5000积分每分钟可以请求30次，"
        "8000积分以上每分钟500次</p><p>本接口单独开权限</p><p>已经停止更新</p>"
        "<p>积分达到6000即可调用</p>",
        ".html",
    )
    claims = catalog._claims(body, "sample")
    assert any(claim.get("row_limit") == 6000 for claim in claims)
    tier = next(claim for claim in claims if "frequency_tiers" in claim)
    assert tier["frequency_per_minute"] == 500
    assert tier["frequency_for_points"] == 15000
    assert tier["frequency_tiers"] == [
        {"minimum_points": 5000, "frequency_per_minute": 30},
        {"minimum_points": 8000, "frequency_per_minute": 500},
    ]
    assert any(claim.get("permission_mode") == "independent" for claim in claims)
    assert any(claim.get("deprecated") is True for claim in claims)
    assert any(claim.get("minimum_points") == 6000 for claim in claims)


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("每天18点更新", "每天18点更新"),
        ("更新时间：15点至17点", "更新时间：15点至17点"),
        ("没有披露时间", ""),
    ],
)
def test_update_description_uses_explicit_schedule(sentence: str, expected: str) -> None:
    """数据更新说明只取明确原文；例如没有发布时间的正文返回空串，不能使用核验日期。"""
    assert catalog._update_description(catalog._body(f"<p>{sentence}</p>", ".html")) == expected


def test_merge_preserves_conflicts_and_independent_isolation() -> None:
    """积分或权限冲突保留证据，独立权限优先隔离；例如 2000 与 5000 冲突后须复核。"""
    entry = CatalogEntry("daily")
    catalog._merge(entry, {"permission_mode": "points", "minimum_points": 2000, "row_limit": 6000})
    catalog._merge(entry, {"minimum_points": 5000, "permission_mode": "independent"})
    catalog._merge(entry, {"deprecated": True})
    assert entry.minimum_points == 2000
    assert entry.permission_mode == "independent"
    assert entry.needs_review
    assert entry.lifecycle == "deprecated"
    assert len(entry.evidence) == 3


def test_entitlement_tables_map_scopes_links_and_account_limits() -> None:
    """总表按分类和链接合并到已发现 API，海外宏观不误归港美股；例如 us_tycr 保持未知。"""
    entries = {
        name: CatalogEntry(name)
        for name in [
            "daily",
            "stk_mins",
            "hk_income",
            "us_income",
            "us_daily",
            "us_tycr",
            "news",
            "linked_api",
        ]
    }
    entries["linked_api"].doc_url = "https://tushare.pro/document/2?doc_id=777"
    table_html = """<p>独立权限接口</p><p>分钟和港美股数据权限不在积分范畴内</p>
        <table><tr><th>积分数</th><th>每分钟频次</th></tr><tr><td>120</td><td>50</td></tr></table>
        <table><tr><th>类型</th><th>包含数据</th></tr>
        <tr><td>港股财报</td><td>财报</td></tr>
        <tr><td>美股财报</td><td>财报</td></tr>
        <tr><td>美股日线</td><td>行情</td></tr>
        <tr><td>新闻资讯</td><td>快讯</td></tr>
        <tr><td>扩展</td><td><a href="/document/2?doc_id=777">详情</a></td></tr></table>"""
    catalog._table_evidence(
        [
            (Path("plain.html"), catalog._body("<p>普通正文</p>", ".html"), ""),
            (
                Path("entitlements.html"),
                catalog._body(table_html, ".html"),
                catalog.ENTITLEMENT_URL,
            ),
        ],
        entries,
    )
    assert entries["daily"].minimum_points == 120
    assert entries["daily"].frequency_per_minute is None
    for name in ("stk_mins", "hk_income", "us_income", "us_daily", "news", "linked_api"):
        assert entries[name].permission_mode == "independent"
    assert entries["us_tycr"].permission_mode == "unknown"


@pytest.mark.parametrize(
    ("filename", "source", "expected"),
    [
        ("doc-27.html", "", "https://tushare.pro/document/2?doc_id=27"),
        ("entitlements.html", "", catalog.ENTITLEMENT_URL),
        ("directory.html", "", catalog.DIRECTORY_URL),
        (
            "sample.md",
            "source_url: https://tushare.pro/document/2?doc_id=28",
            "https://tushare.pro/document/2?doc_id=28",
        ),
        ("sample.md", "source_url: https://tushare.pro/document/2?doc_id=bad", ""),
        ("sample.md", "", ""),
    ],
)
def test_source_url_provenance(filename: str, source: str, expected: str) -> None:
    """来源优先采用白名单标记，其次确定文件名；例如无法识别的 sample.md 返回空来源。"""
    assert catalog._source_url(Path(filename), source) == expected


def test_fetch_failure_manifest_preserves_known_and_missing_documents(tmp_path: Path) -> None:
    """抓取失败既标记旧记录也补充缺失入口；例如 daily 超时和 doc-999 失败都进入债务。"""
    manifest = tmp_path / "index.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "status": "failed",
                    "doc_url": "https://tushare.pro/document/2?doc_id=27",
                    "error": "ReadTimeout",
                },
                {
                    "status": "failed",
                    "doc_url": "https://tushare.pro/document/2?doc_id=999",
                    "error": "HTTPStatusError",
                },
                {"status": "failed", "doc_url": "https://external.example/document/2"},
                {"status": "ok", "doc_url": catalog.DIRECTORY_URL},
                "ignored",
            ]
        ),
        encoding="utf-8",
    )
    entries = {"daily": CatalogEntry("daily", doc_url="https://tushare.pro/document/2?doc_id=27")}
    catalog._merge_fetch_failures([tmp_path], entries, "2026-10-01")
    assert entries["daily"].needs_review
    assert entries["daily"].evidence[0]["error_type"] == "ReadTimeout"
    assert entries["unresolved_doc_999"].needs_review
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="抓取清单必须为列表"):
        catalog._merge_fetch_failures([manifest], {}, "2026-10-01")


def test_sync_catalog_keeps_unknown_interfaces_and_non_api_entries(tmp_path: Path) -> None:
    """离线同步保留未知 API、未解析入口和非 API 交付说明；例如新接口保持 unclassified。"""
    write_document(
        tmp_path,
        "directory.html",
        """<div id="jstree"><ul>
        <li><a href="/document/2?doc_id=10">类别</a><ul>
        <li><a href="/document/2?doc_id=11">已知</a></li>
        <li><a href="/document/2?doc_id=12">新接口</a></li>
        <li><a href="/document/2?doc_id=13">未取得</a></li>
        <li><a href="/document/2?doc_id=14">Tick交付</a></li>
        </ul></li></ul></div><article>数据目录</article>""",
    )
    write_document(
        tmp_path,
        "doc-11.html",
        "<h1>已知</h1><p>接口：known</p><p>需要2000积分</p><p>每天18点更新</p>",
    )
    write_document(tmp_path, "doc-12.html", "<p>接口：new_api</p><p>单次最大100条</p>")
    write_document(tmp_path, "doc-14.html", "<h1>Tick</h1><p>不提供API方式，只提供CSV交付</p>")
    entries = {
        entry.api_name: entry
        for entry in catalog.sync_catalog(
            [tmp_path],
            "2026-10-01",
            {"known": Policy("known", category="market", lifecycle="research")},
        )
    }
    assert "unresolved_doc_10" not in entries
    assert entries["known"].lifecycle == "research"
    assert entries["known"].needs_review
    assert entries["known"].updated_at == "每天18点更新"
    assert entries["new_api"].lifecycle == "unclassified"
    assert entries["new_api"].needs_review
    assert entries["unresolved_doc_13"].needs_review
    assert entries["unresolved_doc_14"].permission_mode == "independent"
    assert not entries["unresolved_doc_14"].needs_review
    assert entries["unresolved_doc_14"].evidence[0]["kind"] == "non_api_delivery"


def test_sync_catalog_markdown_conflicts_and_scope_block(tmp_path: Path) -> None:
    """HTML 与 Markdown 合并保留冲突，不把文档复核当实测阻断；例如 alpha 仍为研究状态。"""
    first = write_document(
        tmp_path, "doc-21.html", "<h1>接口</h1><p>接口：alpha</p><p>2000积分起，单次最大100条</p>"
    )
    second = write_document(
        tmp_path,
        "detail.md",
        "<!-- source_url: https://tushare.pro/document/2?doc_id=21 -->\n"
        "# 接口\n\n接口：alpha\n\n5000积分起，单次最大100条\n",
    )
    blocked = write_document(
        tmp_path, "doc-22.html", "<p>接口：blocked_api</p><p>2000积分起，单次最大100条</p>"
    )
    policies = {
        "alpha": Policy("alpha", lifecycle="research"),
        "blocked_api": Policy("blocked_api", lifecycle="blocked", purpose="非市场数据范围"),
    }
    entries = {
        entry.api_name: entry
        for entry in catalog.sync_catalog([first, second, blocked], "2026-10-01", policies)
    }
    assert entries["alpha"].needs_review
    assert entries["alpha"].lifecycle == "research"
    assert len([e for e in entries["alpha"].evidence if e["kind"] == "discovery"]) == 2
    assert entries["blocked_api"].lifecycle == "blocked"
    assert any(e.get("empirical_unavailable") is False for e in entries["blocked_api"].evidence)


def test_official_snapshot_has_traceable_coverage_and_vip_policies() -> None:
    """真实净化快照可离线发现 250 个 API，保留 Tick 例外；例如 income_vip 门槛为 5000。"""
    fixture_directory = Path(__file__).parents[1] / "fixtures" / "catalog" / "official"
    entries = catalog.sync_catalog([fixture_directory], "2026-10-01")
    indexed = {entry.api_name: entry for entry in entries}
    assert len(entries) == 251
    assert catalog.catalog_summary(entries)["discovered_interfaces"] == 250
    assert indexed["income"].minimum_points == 2000
    assert indexed["income_vip"].minimum_points == 5000
    assert indexed["income_vip"].lifecycle == "research"
    assert indexed["income_vip"].needs_review
    assert indexed["rt_k"].permission_mode == "independent"
    assert indexed["stk_factor_pro"].frequency_per_minute == 500
    assert indexed["broker_recommend"].minimum_points == 6000
    assert indexed["us_daily"].needs_review
    assert indexed["unresolved_doc_314"].evidence[0]["kind"] == "non_api_delivery"


@pytest.mark.parametrize(
    "example",
    [
        "ts.set_token('DOC_SAMPLE_SECRET')",
        "ts.pro_api('DOC_SAMPLE_SECRET')",
        "ts.pro_api(token='DOC_SAMPLE_SECRET')",
    ],
)
def test_sanitize_document_redacts_credential_arguments(example: str) -> None:
    """凭据示例实参均须脱敏，包含关键字调用；例如 token 参数的哨兵字符串不得进入快照。"""
    source = (
        f"<div class='content col-md-9'><p>接口：daily</p><pre>{example}</pre></div>"
        "<script>tracking()</script>"
    )
    sanitized = catalog._sanitize_document(source)
    assert "DOC_SAMPLE_SECRET" not in sanitized
    assert "REDACTED" in sanitized
    assert "tracking" not in sanitized


def test_sanitize_document_handles_navigation_and_plain_fragments() -> None:
    """目录导航保留且普通片段能净化，详情不携带重复侧栏；例如登录表单始终删除。"""
    directory = "<div id='jstree'><a href='/document/2?doc_id=27'>日线</a></div><form>登录</form>"
    assert "jstree" in catalog._sanitize_document(directory)
    assert "登录" not in catalog._sanitize_document(directory)
    plain = "<article><h1>正文</h1><script>tracking()</script></article>"
    assert "正文" in catalog._sanitize_document(plain)
    assert "tracking" not in catalog._sanitize_document(plain)
    detailed = directory + "<div class='content col-md-9'><p>接口：daily</p></div>"
    assert "jstree" not in catalog._sanitize_document(detailed)


def test_fetch_one_isolates_http_status_and_timeout() -> None:
    """文档请求不跟随跳转，HTTP 错误与超时返回类型；例如 403 不被记作成功文档。"""
    url = catalog.DIRECTORY_URL
    with patch.object(catalog.httpx, "get", autospec=True) as get:
        get.return_value = httpx.Response(
            200, text="<p>文档</p>", request=httpx.Request("GET", url)
        )
        result = catalog._fetch_one(url)
        assert result[0] == url and "文档" in result[1] and result[2] == ""
        get.assert_called_once_with(url, timeout=30, follow_redirects=False, trust_env=False)
    with patch.object(catalog.httpx, "get", autospec=True) as get:
        get.return_value = httpx.Response(403, request=httpx.Request("GET", url))
        assert catalog._fetch_one(url) == (url, "", "HTTPStatusError")
        assert get.call_count == 1
    with patch.object(catalog.httpx, "get", autospec=True) as get:
        get.side_effect = httpx.ReadTimeout("controlled timeout")
        assert catalog._fetch_one(url) == (url, "", "ReadTimeout")
        assert get.call_count == 1
    with patch.object(catalog.httpx, "get", autospec=True) as get:
        assert catalog._fetch_one("https://api.tushare.pro") == (
            "https://api.tushare.pro",
            "",
            "InvalidDocumentURL",
        )
        get.assert_not_called()


def test_fetch_documentation_tracks_links_failures_and_hashes(tmp_path: Path) -> None:
    """爬取只跟随文档链接，成功文件可核验且失败不丢失；例如 doc-28 超时仍在清单。"""
    successful = "https://tushare.pro/document/2?doc_id=27"
    failing = "https://tushare.pro/document/2?doc_id=28"
    pages = {
        catalog.DIRECTORY_URL: (
            "<article><a href='/document/2?doc_id=27'>行情</a>"
            "<a href='/document/2?doc_id=28'>失败</a>"
            "<a href='https://api.tushare.pro'>禁止</a></article>"
        ),
        catalog.ENTITLEMENT_URL: "<article>独立权限接口</article>",
        successful: (
            "<article><p>接口：daily</p><a href='/document/2?doc_id=27'>自身</a>"
            "<a href='/document/2?doc_id=28'>同批失败页</a></article>"
        ),
    }

    def respond(url: str, **kwargs: Any) -> httpx.Response:
        """按确定的公开文档映射返回响应；输入 doc-28 返回可控超时，其余返回对应 HTML。"""
        assert kwargs == {"timeout": 30, "follow_redirects": False, "trust_env": False}
        if url == failing:
            raise httpx.ReadTimeout("controlled timeout")
        return httpx.Response(200, text=pages[url], request=httpx.Request("GET", url))

    with patch.object(catalog.httpx, "get", autospec=True, side_effect=respond) as get:
        paths = catalog.fetch_documentation(tmp_path)
        assert get.call_count == 4
    assert tmp_path / "index.json" in paths
    manifest = json.loads((tmp_path / "index.json").read_text())
    assert len(manifest) == 4
    failed = next(item for item in manifest if item["doc_url"] == failing)
    assert failed["status"] == "failed" and failed["error"] == "ReadTimeout"
    for item in manifest:
        if item["status"] == "ok":
            path = tmp_path / item["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    entries = {entry.api_name: entry for entry in catalog.sync_catalog(paths, "2026-10-01", {})}
    assert entries["unresolved_doc_28"].needs_review
    assert "daily" in entries


def test_catalog_summary_separates_unresolved_documents() -> None:
    """摘要不会把未解析文档算作 API，也不会以空集合宣称完成；例如空目录发现数为零。"""
    assert catalog.catalog_summary([])["discovered_interfaces"] == 0
    entries = [CatalogEntry("daily"), CatalogEntry("unresolved_doc_1", needs_review=True)]
    summary = catalog.catalog_summary(entries)
    assert summary["discovered_interfaces"] == 1
    assert len(summary["unresolved_documents"]) == 1
    assert summary["needs_review"] == ["unresolved_doc_1"]
    assert summary["lifecycle_counts"] == {"unclassified": 2}
    assert "不代表官方全部接口" in summary["coverage_claim"]


def test_schema_summary_exactly_matches_discovered_interfaces() -> None:
    """场景：全量 schema 摘要与发现总账对照；预期：250 个名称完全一致且 pro_bar 可审计。

    示例：pro_bar 来自 doc_id=109，9 个输入字段保留，未披露输出为 []。
    """
    root = Path(__file__).parents[1] / "fixtures" / "catalog"
    schemas = json.loads((root / "schema-summary.json").read_text())
    names = [item["api_name"] for item in schemas]
    discovered = {
        item.api_name
        for item in catalog.sync_catalog([root / "official"], "2026-10-01")
        if not item.api_name.startswith("unresolved_")
    }
    assert len(names) == len(set(names)) == 250
    assert set(names) == discovered == set(catalog.load_policies())
    item = next(item for item in schemas if item["api_name"] == "pro_bar")
    assert item["doc_url"] == "https://tushare.pro/document/2?doc_id=109"
    assert [field["name"] for field in item["input_fields"]] == [
        "ts_code",
        "start_date",
        "end_date",
        "asset",
        "adj",
        "freq",
        "ma",
        "factors",
        "adjfactor",
    ]
    assert [field["required"] for field in item["input_fields"]] == ["Y"] + ["N"] * 8
    assert item["output_fields"] == []
