default_step_rules = {
    "data": {
        "extract": {
            "filepath": {
                "required": True
            },
            "target": {
                "required": True
            }
        },
        "transform": {
            "normalize": {
                "features": {
                    "module": {
                        "required": True
                    },
                    "target": {
                        "module": {
                            "required": True
                        }
                    }
                }
            }
        },
        "load": {
            "path": {
                "required": True
            }
        }
    },
    "model": {
        "extract": {
            "filepath": {
                "required": True
            }
        },
        "transform": {
            "fit": {
                "cross_validation": {
                    "strategy": {
                        "required": True
                    }
                }
            },
            "predict": {
                "path": {
                    "required": True
                }
            }
        },
        "load": {
            "path": {
                "required": True
            }
        }
    },
    "benchmark": None
}
