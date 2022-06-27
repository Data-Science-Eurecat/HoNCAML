from src.tools import utils


def test_ensure_input_list_nolist():
    """Test ensure_input_list method."""
    # No list
    obj = 1
    expected = [1]
    result = utils.ensure_input_list(obj)
    assert expected == result
    # List
    obj = [1]
    expected = [1]
    result = utils.ensure_input_list(obj)
    # None object
    obj = None
    expected = []
    result = utils.ensure_input_list(obj)
    assert expected == result
