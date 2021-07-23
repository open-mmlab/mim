from mim.commands.list import list_package


def get_installed_package(ctx, args, incomplete):
    pkgs = []
    for pkg, _, _ in list_package():
        pkgs.append(pkg)
    return pkgs


def get_downstream_package(ctx, args, incomplete):
    pkgs = []
    for pkg, _, _ in list_package():
        if pkg == 'mmcv' or pkg == 'mmcv-full':
            continue
        pkgs.append(pkg)
    return pkgs


def get_official_package(ctx=None, args=None, incomplete=None):
    return [
        'mmcls',
        'mmdet',
        'mmdet3d',
        'mmseg',
        'mmaction',
        'mmtrack',
        'mmpose',
        'mmedit',
        'mmocr',
        'mmgen',
    ]
