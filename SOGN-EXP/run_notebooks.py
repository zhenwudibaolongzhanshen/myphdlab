"""批量运行指定数据集的所有 notebook

用法:
  python run_notebooks.py --dataset california_housing
  python run_notebooks.py --dataset beijing_pm25 --methods sogn,standard
"""

import json, os, sys, traceback, io, argparse

# 修复 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

METHODS = ["sogn", "standard", "mc_dropout", "deep_ensemble", "deep_evidential",
           "selectivenet", "conformal"]


def run_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_lines = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            code_lines.append(source)
            code_lines.append('\n')

    full_code = '\n'.join(code_lines)

    nb_dir = os.path.dirname(os.path.abspath(nb_path))
    old_cwd = os.getcwd()
    os.chdir(nb_dir)

    try:
        exec(compile(full_code, nb_path, 'exec'), {'__name__': '__main__'})
        print(f'[OK] {os.path.basename(nb_path)}')
        return True
    except Exception:
        print(f'[FAIL] {os.path.basename(nb_path)}')
        traceback.print_exc()
        return False
    finally:
        os.chdir(old_cwd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--methods', type=str, default=None,
                        help='comma-separated method names, default=all')
    args = parser.parse_args()

    methods = args.methods.split(',') if args.methods else METHODS
    base = os.path.dirname(os.path.abspath(__file__))

    for method in methods:
        nb_path = os.path.join(base, args.dataset, f'{method}.ipynb')
        if not os.path.exists(nb_path):
            print(f'[SKIP] {nb_path} not found')
            continue
        sep = '=' * 50
        print(f'\n{sep}\nRunning: {args.dataset}/{method}\n{sep}')
        run_notebook(nb_path)


if __name__ == '__main__':
    main()
