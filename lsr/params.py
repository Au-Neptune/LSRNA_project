import os
import glob
import yaml
import csv

from .models import make

# import 所有完整模型需要的 components
from . import liif
from . import swinir
from . import mlp
from . import hat
from . import drct
from . import dat

def count_params(module, trainable_only=False):
    if module is None:
        return 0

    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    return sum(p.numel() for p in module.parameters())


def get_submodule_if_exists(model, names):
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def count_model_params_from_yaml(yaml_path, trainable_only=False):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    model_spec = config['model']
    model_name = model_spec['name']

    model = make(model_spec)

    encoder = get_submodule_if_exists(
        model,
        ['encoder', 'encoder_net', 'backbone', 'feat_encoder']
    )

    imnet = get_submodule_if_exists(
        model,
        ['imnet', 'mlp']
    )

    total_params = count_params(model, trainable_only=trainable_only)
    encoder_params = count_params(encoder, trainable_only=trainable_only)
    imnet_params = count_params(imnet, trainable_only=trainable_only)

    other_params = total_params - encoder_params - imnet_params

    encoder_spec = model_spec['args'].get('encoder_spec', {})
    imnet_spec = model_spec['args'].get('imnet_spec', {})

    return {
        'yaml': yaml_path,
        'model_name': model_name,
        'encoder_name': encoder_spec.get('name', 'none'),
        'imnet_name': imnet_spec.get('name', 'none'),

        'total_params': total_params,
        'total_params_M': total_params / 1e6,

        'encoder_params': encoder_params,
        'encoder_params_M': encoder_params / 1e6,

        'imnet_params': imnet_params,
        'imnet_params_M': imnet_params / 1e6,

        'other_params': other_params,
        'other_params_M': other_params / 1e6,
    }


def count_many_yamls(yaml_dir, trainable_only=False):
    yaml_paths = sorted(
        glob.glob(os.path.join(yaml_dir, '*.yaml')) +
        glob.glob(os.path.join(yaml_dir, '*.yml'))
    )

    results = []

    for yaml_path in yaml_paths:
        try:
            result = count_model_params_from_yaml(
                yaml_path,
                trainable_only=trainable_only
            )

            results.append(result)

            print(
                f"{os.path.basename(yaml_path):40s} | "
                f"model={result['model_name']:10s} | "
                f"encoder={result['encoder_name']:15s} | "
                f"total={result['total_params_M']:.3f}M | "
                f"encoder={result['encoder_params_M']:.3f}M | "
                f"imnet={result['imnet_params_M']:.3f}M | "
                f"other={result['other_params_M']:.3f}M"
            )

        except Exception as e:
            print(f"[ERROR] {yaml_path}")
            print(f"        {type(e).__name__}: {e}")

    return results


if __name__ == '__main__':
    yaml_dir = './lsr'   # 依照你 YAML 位置修改

    results = count_many_yamls(
        yaml_dir,
        trainable_only=False
    )

    with open('model_params.csv', 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'yaml',
                'model_name',
                'encoder_name',
                'imnet_name',
                'total_params',
                'total_params_M',
                'encoder_params',
                'encoder_params_M',
                'imnet_params',
                'imnet_params_M',
                'other_params',
                'other_params_M',
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print('Saved to model_params.csv')