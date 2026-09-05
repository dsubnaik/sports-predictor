from baseball.ui.report_view import build_projection_rows, filter_pitcher_lines


def test_filter_pitcher_lines_matches_case_insensitive_search_without_mutating():
    lines = [
        {"pitcher": "Tarik Skubal", "line": 6.5},
        {"pitcher": "Paul Skenes", "line": 7.5},
    ]

    result = filter_pitcher_lines(lines, "sku")

    assert result == [{"pitcher": "Tarik Skubal", "line": 6.5}]
    assert lines == [
        {"pitcher": "Tarik Skubal", "line": 6.5},
        {"pitcher": "Paul Skenes", "line": 7.5},
    ]


def test_filter_pitcher_lines_blank_search_returns_copied_rows():
    lines = [{"pitcher": "Paul Skenes", "line": 7.5}]

    result = filter_pitcher_lines(lines, " ")
    result[0]["line"] = 8.5

    assert lines[0]["line"] == 7.5


def test_build_projection_rows_attaches_projection_by_pitcher_name():
    lines = [
        {"pitcher": "Tarik Skubal", "line": 6.5},
        {"pitcher": "Paul Skenes", "line": 7.5},
    ]

    result = build_projection_rows(lines, {"Tarik Skubal": 7.1})

    assert result[0]["predicted_ks"] == 7.1
    assert result[1]["predicted_ks"] is None
    assert "predicted_ks" not in lines[0]
