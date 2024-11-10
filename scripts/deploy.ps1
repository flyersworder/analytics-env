#Requires -Version 5.1
#Requires -RunAsAdministrator

param(
    [switch]$Force,
    [string]$LogPath = "setup.log"
)

$ErrorActionPreference = "Stop"

# Initialize logging
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet('Info', 'Warning', 'Error')]
        [string]$Level = 'Info'
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "$timestamp - [$Level] $Message"
    $logMessage | Out-File -Append -FilePath $LogPath

    switch ($Level) {
        'Info' { Write-Host $Message }
        'Warning' { Write-Host $Message -ForegroundColor Yellow }
        'Error' { Write-Host $Message -ForegroundColor Red }
    }
}

# Function to validate prerequisites
function Test-Prerequisites {
    if (!(Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Log "Git is not installed or not found in PATH." -Level 'Error'
        return $false
    }
    return $true
}

# Function to refresh environment variables
function Update-Environment {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    # Verify uv is now accessible
    if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
        return $false
    }
    return $true
}

# Function to securely store credentials
function Set-SecureCredentials {
    param(
        [string]$WindowsUser,
        [SecureString]$WindowsPassword,
        [string]$UnixUser,
        [SecureString]$UnixPassword
    )

    try {
        # Store credentials using cmdkey
        $windowsCred = New-Object PSCredential($WindowsUser, $WindowsPassword)
        $unixCred = New-Object PSCredential($UnixUser, $UnixPassword)

        cmdkey /add:dive-prod.infineon.com /user:$WindowsUser /pass:$($windowsCred.GetNetworkCredential().Password)
        cmdkey /add:cbdp-impala-prod.muc.infineon.com /user:$UnixUser /pass:$($unixCred.GetNetworkCredential().Password)

        return $true
    }
    catch {
        Write-Log "Failed to store credentials: $_" -Level 'Error'
        return $false
    }
}

# Main script execution
try {
    Write-Log "Starting environment setup..."

    # Check prerequisites
    if (!(Test-Prerequisites)) {
        throw "Prerequisites check failed"
    }

    # Repository setup
    $repoDir = "analytics-topics"
    if (Test-Path -Path $repoDir) {
        if ($Force) {
            Write-Log "Force flag set. Removing existing repository..." -Level 'Warning'
            Remove-Item -Path $repoDir -Recurse -Force
        } else {
            Write-Log "Repository already exists. Use -Force to override." -Level 'Warning'
            exit 0
        }
    }
    Write-Log "Cloning the repository..."
    try {
        git clone git@github.com:example-org/analytics-project.git
    } catch {
        Write-Log "Failed to clone repository: $_" -Level 'Error'
        throw
    }

    Set-Location $repoDir

    # UV installation and setup
    if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Log "Installing uv..."
        $uvInstallScript = Join-Path $env:TEMP "uv_install.ps1"
        Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile $uvInstallScript
        PowerShell -ExecutionPolicy Bypass -File $uvInstallScript
        Remove-Item $uvInstallScript

        # Refresh environment variables
        Write-Log "Refreshing environment variables..."
        if (!(Update-Environment)) {
            # Restart the script in a new PowerShell session to pick up updated PATH
            Write-Log "Restarting script in new PowerShell session to update PATH..." -Level 'Warning'
            $scriptPath = $MyInvocation.MyCommand.Path
            $arguments = @()
            foreach ($param in $MyInvocation.BoundParameters.GetEnumerator()) {
                if ($param.Value -is [switch]) {
                    if ($param.Value.IsPresent) {
                        $arguments += "-$($param.Key)"
                    }
                } else {
                    $arguments += "-$($param.Key) `"$($param.Value)`""
                }
            }
            Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$scriptPath`" $arguments" -Wait
            exit 0
        }
    }

    # Python setup
    if (!(uv python list | Select-String -Pattern "3.12")) {
        Write-Log "Installing Python 3.12..."
        uv python install 3.12
    }

    Write-Log "Setting up the Python environment..."
    uv sync

    # DVC setup
    if (!(Test-Path -Path ".dvc")) {
        Write-Log "Initializing DVC..."
        uv run dvc init
    }

    # Pre-commit hooks
    Write-Log "Installing pre-commit hooks..."
    uv run pre-commit install

    # Configuration setup
    $configFilePath = Join-Path $repoDir "db_config.ini"
    $configBackupPath = "$configFilePath.backup"

    if (Test-Path $configFilePath) {
        if ($Force) {
            Write-Log "Backing up existing config..." -Level 'Warning'
            Copy-Item $configFilePath $configBackupPath -Force
        } else {
            Write-Log "Configuration file already exists. Use -Force to override." -Level 'Warning'
            exit 0
        }
    }

    # Credential collection and storage
    Write-Log "Collecting credentials..."
    $windowsUser = Read-Host -Prompt "Enter your Windows login username"
    $windowsPassword = Read-Host -Prompt "Enter your Windows login password" -AsSecureString
    $unixUser = Read-Host -Prompt "Enter your UNIX login username"
    $unixPassword = Read-Host -Prompt "Enter your UNIX login password" -AsSecureString

    # Store credentials securely
    if (!(Set-SecureCredentials -WindowsUser $windowsUser -WindowsPassword $windowsPassword `
                -UnixUser $unixUser -UnixPassword $unixPassword)) {
        throw "Failed to store credentials securely"
    }

    # Create configuration using stored credentials
    $configContent = @"
[dive]
host = dive-prod.infineon.com
port = 9996
database = vdb_std
user = $windowsUser
use_windows_auth = true

[impala_db]
host = cbdp-impala-prod.muc.infineon.com
database = user_$unixUser
port = 21050
user = $unixUser
use_kerberos = true
"@

    $configContent | Out-File -FilePath $configFilePath -Encoding UTF8

    # Validate configuration
    if (Test-Path $configFilePath) {
        Write-Log "Configuration file created successfully"
        Write-Log "Configuration file location: $configFilePath"
    } else {
        throw "Failed to create configuration file"
    }

    Write-Log "Environment setup complete!"
}
catch {
    Write-Log "Setup failed: $_" -Level 'Error'
    exit 1
}
