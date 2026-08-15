from klas_model.collectors.afd import analyze_afd_text


def test_afd_flags_monsoon_outflow():
    result = analyze_afd_text(
        "Monsoonal moisture will support afternoon thunderstorms. Strong outflow winds are possible near Las Vegas."
    )
    assert result["risk"] == "HIGH"
    assert "outflow" in result["high_terms"]
    assert "monsoon" in result["convective_terms"]


def test_afd_can_remain_low_when_no_signal():
    result = analyze_afd_text("Hot and dry conditions continue with clear skies through this afternoon.")
    assert result["risk"] == "LOW"
