def test_vaca_package_imports():
    import vaca

    assert vaca.__version__.startswith("0.1.0")
