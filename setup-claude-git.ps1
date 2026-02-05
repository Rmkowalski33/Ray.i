# Claude Code Git Bash Setup Script (No Admin Required)
# Save this as: setup-claude-git.ps1

Write-Host "=== Claude Code Git Bash Configuration ===" -ForegroundColor Cyan
Write-Host ""

# Function to find Git Bash
function Find-GitBash {
    $commonPaths = @(
        "C:\Program Files\Git\bin\bash.exe",
        "C:\Program Files (x86)\Git\bin\bash.exe",
        "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe",
        "$env:ProgramFiles\Git\bin\bash.exe"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    
    # Try to find Git using where.exe
    $gitPath = (Get-Command git.exe -ErrorAction SilentlyContinue).Source
    if ($gitPath) {
        $bashPath = $gitPath -replace 'cmd\\git.exe$', 'bin\bash.exe' -replace 'bin\\git.exe$', 'bin\bash.exe'
        if (Test-Path $bashPath) {
            return $bashPath
        }
    }
    
    return $null
}

# Step 1: Find Git Bash
Write-Host "Step 1: Looking for Git Bash installation..." -ForegroundColor Yellow
$gitBashPath = Find-GitBash

if ($gitBashPath) {
    Write-Host "✓ Found Git Bash at: $gitBashPath" -ForegroundColor Green
} else {
    Write-Host "✗ Git Bash not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "You may need to request Git installation from IT, or install it to your user directory." -ForegroundColor Yellow
    exit 1
}

# Step 2: Set environment variable (User scope - no admin needed)
Write-Host ""
Write-Host "Step 2: Setting CLAUDE_CODE_GIT_BASH_PATH environment variable..." -ForegroundColor Yellow

try {
    # Set for current user (persistent, no admin required)
    [Environment]::SetEnvironmentVariable("CLAUDE_CODE_GIT_BASH_PATH", $gitBashPath, "User")
    
    # Set for current session
    $env:CLAUDE_CODE_GIT_BASH_PATH = $gitBashPath
    
    Write-Host "✓ Environment variable set successfully!" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to set environment variable: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Verify
Write-Host ""
Write-Host "Step 3: Verifying configuration..." -ForegroundColor Yellow
$verifyUser = [Environment]::GetEnvironmentVariable("CLAUDE_CODE_GIT_BASH_PATH", "User")
$verifySession = $env:CLAUDE_CODE_GIT_BASH_PATH

if ($verifyUser -eq $gitBashPath -and $verifySession -eq $gitBashPath) {
    Write-Host "✓ Configuration verified!" -ForegroundColor Green
} else {
    Write-Host "✗ Verification failed" -ForegroundColor Red
    Write-Host "User env: $verifyUser" -ForegroundColor Gray
    Write-Host "Session env: $verifySession" -ForegroundColor Gray
}

# Step 4: Instructions
Write-Host ""
Write-Host "=== Next Steps ===" -ForegroundColor Cyan
Write-Host "1. Close and reopen VS Code completely" -ForegroundColor White
Write-Host "2. Open a new PowerShell terminal in VS Code" -ForegroundColor White
Write-Host "3. Run: claude" -ForegroundColor White
Write-Host ""
Write-Host "Current Git Bash path: $gitBashPath" -ForegroundColor Gray