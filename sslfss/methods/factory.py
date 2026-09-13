from sslfss.methods.base import FewShotMethod


def get_method(name: str, method_cfg=None) -> FewShotMethod:
    from sslfss.methods.maml_method import MAMLMethod
    from sslfss.methods.panet_method import PANetMethod
    from sslfss.methods.r2d2_method import R2D2Method
    from sslfss.methods.alpnet_method import ALPNetMethod

    options = {
        "maml":   lambda: MAMLMethod(method_cfg),
        "panet":  lambda: PANetMethod(method_cfg),
        "r2d2":   lambda: R2D2Method(method_cfg),
        "alpnet": lambda: ALPNetMethod(method_cfg),
    }
    if name not in options:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(options)}")
    return options[name]()
