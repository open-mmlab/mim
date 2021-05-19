from mim.commands.list import list_package


def test_list():
    # mim list
    target = ('mmcls', '0.11.0',
              'https://github.com/open-mmlab/mmclassification.git')
    result = list_package()
    assert target in result
