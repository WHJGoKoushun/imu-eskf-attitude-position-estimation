$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
# Python 3.12 为已验证版本；锁定依赖要求 Python >=3.12,<4。
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$versionCheck = @'
import sys
print('Python', sys.version.split()[0])
if not (3, 12) <= sys.version_info[:2] < (4, 0):
    sys.exit('Locked dependencies require Python >=3.12,<4. Python 3.12 is recommended.')
if sys.version_info[:2] != (3, 12):
    print('WARNING: only Python 3.12 has been validated; trying this version with all checks enabled.')
'@
if (Test-Path -LiteralPath $projectPython) {
    $runner = $projectPython
} else {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    $bootstrapCommand = $null
    $bootstrapArguments = @()
    if ($pyLauncher) {
        # 探测只用于选择解释器，不把缺少 3.12 当作最终失败。
        & py -3.12 -c $versionCheck
        if ($LASTEXITCODE -eq 0) {
            $bootstrapCommand = 'py'
            $bootstrapArguments = @('-3.12')
        } else {
            & py -3 -c $versionCheck
            if ($LASTEXITCODE -eq 0) {
                $bootstrapCommand = 'py'
                $bootstrapArguments = @('-3')
            }
        }
    }
    if (-not $bootstrapCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c $versionCheck
        if ($LASTEXITCODE -eq 0) { $bootstrapCommand = 'python' }
    }
    if (-not $bootstrapCommand) {
        throw '未找到满足锁定依赖的 Python >=3.12,<4。建议安装已验证的 Python 3.12 并加入 PATH。'
    }
    & $bootstrapCommand @bootstrapArguments -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw '创建虚拟环境失败，请检查 Python 的 venv 组件及目录权限。' }
    $runner = $projectPython
}
& $runner -c $versionCheck
if ($LASTEXITCODE -ne 0) { throw '项目 .venv 不满足 Python >=3.12,<4；请改名保留旧环境，并用 Python 3.12 重新建立。' }
# 如果首次网络安装失败，下次仍会补齐依赖；已满足版本时不会重复下载。
& $runner -m pip install --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw '锁定依赖安装失败。请查看上方 pip 信息，检查网络或解释器/平台兼容性；建议使用 Python 3.12。' }
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& $runner validate.py --output results/validation
if ($LASTEXITCODE -ne 0) { throw '数值验证失败，已停止。' }
& $runner validate_pipeline.py --output results/validation/pipeline_checks.json
if ($LASTEXITCODE -ne 0) { throw '数据流程验证失败，已停止。' }
& $runner run.py --comparisons
if ($LASTEXITCODE -ne 0) { throw '真实数据实验失败，请查看错误信息。' }
& $runner build_report.py
if ($LASTEXITCODE -ne 0) { throw '报告生成失败，请查看错误信息。' }
Write-Host '全部完成。请打开 REPORT.pdf 和 results/figures。'
