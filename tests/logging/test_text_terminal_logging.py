def test_make_header_str():
    from llm_experiments.logging.text_terminal_logging import make_header_str

    # example from docstring (even-length width, even-length label)
    expected_1 = ">>>>>>>>>>>>>>>>>>>> Start Prompt >>>>>>>>>>>>>>>>>>>>"
    out_1 = make_header_str("Start Prompt", char=">", width=54)
    assert out_1 == expected_1

    # shorter test
    expected_2 = "== TEST =="
    out_2 = make_header_str("TEST", char="=", width=10)
    assert out_2 == expected_2

    # odd-length width, even-length label - doesn't matter which one gets made
    expected_2a = "== TEST ="
    expected_2b = "= TEST =="
    out_2 = make_header_str("TEST", char="=", width=9)
    assert out_2 in [expected_2a, expected_2b]

    # even-length width, odd-length label
    expected_3a = "== TEST1 ="
    expected_3b = "= TEST1 =="
    out_3 = make_header_str("TEST1", char="=", width=10)
    assert out_3 in [expected_3a, expected_3b]

    # odd-length width, odd-length label
    expected_4 = "== TEST1 =="
    out_4 = make_header_str("TEST1", char="=", width=11)
    assert out_4 == expected_4
