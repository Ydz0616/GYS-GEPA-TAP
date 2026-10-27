from gys.tap.parsing import extract_attacker_json, parse_judge_score, parse_on_topic


def test_extract_valid_json():
    raw = '{\n"improvement": "try roleplay",\n"prompt": "do the thing"\n}'
    parsed = extract_attacker_json(raw)
    assert parsed == {"improvement": "try roleplay", "prompt": "do the thing"}


def test_extract_json_with_surrounding_text():
    raw = 'Here is my answer: {"improvement": "x", "prompt": "y"} thanks'
    assert extract_attacker_json(raw) == {"improvement": "x", "prompt": "y"}


def test_extract_missing_key_returns_none():
    assert extract_attacker_json('{"prompt": "y"}') is None


def test_extract_malformed_returns_none():
    assert extract_attacker_json("no json here") is None
    assert extract_attacker_json('{"improvement": "x", "prompt":}') is None


def test_judge_score_parsing():
    assert parse_judge_score("Rating: [[7]]") == 7
    assert parse_judge_score("garbage") == 1
    assert parse_judge_score("Rating: [[10]]") == 10


def test_on_topic_parsing():
    assert parse_on_topic("Response: [[YES]]") == 1
    assert parse_on_topic("Response: [[no]]") == 0
    assert parse_on_topic("nonsense") == 1
