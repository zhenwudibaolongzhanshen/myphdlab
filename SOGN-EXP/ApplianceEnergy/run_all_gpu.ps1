param(
    [string]$Python = (Join-Path $PSScriptRoot '..\..\..\SOGN\.venv\Scripts\python.exe')
)

$notebooks = @(
    'standard.ipynb',
    'mc_dropout.ipynb',
    'deep_ensemble.ipynb',
    'deep_evidential.ipynb',
    'conformal.ipynb',
    'selectivenet.ipynb',
    'sogn.ipynb'
)

foreach ($notebook in $notebooks) {
    & $Python -m jupyter nbconvert --to notebook --execute $notebook `
        --output ($notebook -replace '\.ipynb$', '.executed.ipynb') `
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=sogn-gpu
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $Python evaluate_results.py
& $Python plot_coverage.py
