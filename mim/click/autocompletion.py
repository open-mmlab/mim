from mim.utils import read_installation_records


def get_installed_package(ctx, args, incomplete):
    pkgs = []
    for pkg, _, _ in read_installation_records():
        pkgs.append(pkg)
    return pkgs


def get_downstream_package(ctx, args, incomplete):
    pkgs = []
    for pkg, _, _ in read_installation_records():
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
