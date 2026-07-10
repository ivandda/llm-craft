param(
  [string]$Project = "nlp2026-498021",
  [string]$Vm = "llm-craft-all-in-one",
  [string]$Machine = "g2-standard-4",
  [string[]]$Zones = @(
    "us-central1-c",
    "us-central1-a",
    "us-central1-b",
    "us-west1-a",
    "us-west1-b",
    "us-west1-c",
    "us-east1-b",
    "us-east1-c",
    "us-east1-d",
    "us-east4-a",
    "us-east4-c"
  ),
  [string]$Tag = "llm-craft-all-in-one",
  [string]$RepoUrl = "https://github.com/ivandda/llm-craft.git",
  [string]$Branch = "dev",
  [string]$QwenApiKey = "dev-qwen-key",
  [string]$VertexProject = "",
  [string]$VertexLocation = "us-central1",
  [string]$VertexModel = "gemini-2.5-flash",
  [string]$VertexServiceAccountPath = "",
  [string]$BootDiskSize = "200GB",
  [string]$BootDiskType = "pd-ssd",
  [switch]$SkipBootstrap
)

$ErrorActionPreference = "Stop"

function ConvertTo-BashSingleQuoted {
  param([string]$Value)
  return "'" + ($Value -replace "'", "'\''") + "'"
}

function Invoke-Gcloud {
  & gcloud.cmd @args
  if ($LASTEXITCODE -ne 0) {
    throw "gcloud failed: $($args -join ' ')"
  }
}

function Get-ExistingZone {
  $zone = & gcloud.cmd compute instances list `
    --project $Project `
    --filter "name=($Vm)" `
    --format "value(zone.basename())"

  if ($LASTEXITCODE -ne 0) {
    throw "Could not query existing VM."
  }

  return ($zone | Select-Object -First 1)
}

function Ensure-FirewallRule {
  param(
    [string]$Name,
    [string]$Allow,
    [string]$SourceRanges,
    [string]$Description
  )

  & gcloud.cmd compute firewall-rules describe $Name --project $Project *> $null
  if ($LASTEXITCODE -eq 0) {
    Invoke-Gcloud compute firewall-rules update $Name `
      --project $Project `
      --source-ranges $SourceRanges `
      --target-tags $Tag
    return
  }

  Invoke-Gcloud compute firewall-rules create $Name `
    --project $Project `
    --allow $Allow `
    --source-ranges $SourceRanges `
    --target-tags $Tag `
    --description $Description
}

function Wait-ForSsh {
  param([string]$Zone)

  $deadline = (Get-Date).AddMinutes(8)
  do {
    Start-Sleep -Seconds 10
    & gcloud.cmd compute ssh $Vm `
      --zone $Zone `
      --project $Project `
      --command "echo ssh-ready" `
      --quiet

    if ($LASTEXITCODE -eq 0) {
      return
    }
  } while ((Get-Date) -lt $deadline)

  throw "SSH did not become ready before timeout."
}

function Invoke-Bootstrap {
  param([string]$Zone)

  $bootstrapPath = Join-Path $PSScriptRoot "..\vm\bootstrap_all_in_one.sh"
  $bootstrapPath = (Resolve-Path $bootstrapPath).Path
  $resolvedVertexProject = if ($VertexProject) { $VertexProject } else { $Project }
  $remoteCredentialsPath = ""

  Invoke-Gcloud compute scp `
    --zone $Zone `
    --project $Project `
    $bootstrapPath `
    "${Vm}:~/bootstrap_all_in_one.sh"

  if ($VertexServiceAccountPath) {
    $localCredentialsPath = (Resolve-Path $VertexServiceAccountPath).Path
    $remoteCredentialsPath = "~/.config/llm-craft/vertex-service-account.json"

    Invoke-Gcloud compute ssh $Vm `
      --zone $Zone `
      --project $Project `
      --command "mkdir -p ~/.config/llm-craft && chmod 700 ~/.config/llm-craft"

    Invoke-Gcloud compute scp `
      --zone $Zone `
      --project $Project `
      $localCredentialsPath `
      "${Vm}:${remoteCredentialsPath}"

    Invoke-Gcloud compute ssh $Vm `
      --zone $Zone `
      --project $Project `
      --command "chmod 600 ${remoteCredentialsPath}"
  }

  $envParts = @(
    "REPO_URL=$(ConvertTo-BashSingleQuoted $RepoUrl)",
    "REPO_BRANCH=$(ConvertTo-BashSingleQuoted $Branch)",
    "QWEN_API_KEY=$(ConvertTo-BashSingleQuoted $QwenApiKey)",
    "GOOGLE_CLOUD_PROJECT=$(ConvertTo-BashSingleQuoted $resolvedVertexProject)",
    "VERTEX_LOCATION=$(ConvertTo-BashSingleQuoted $VertexLocation)",
    "VERTEX_MODEL=$(ConvertTo-BashSingleQuoted $VertexModel)"
  )

  if ($remoteCredentialsPath) {
    $envParts += 'GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/llm-craft/vertex-service-account.json"'
    $envParts += "VERTEX_USE_GCE_METADATA='false'"
  } else {
    $envParts += "VERTEX_USE_GCE_METADATA='true'"
  }

  $bootstrapEnv = $envParts -join " "
  $remoteCommand = @"
chmod +x ~/bootstrap_all_in_one.sh
$bootstrapEnv bash ~/bootstrap_all_in_one.sh
"@

  & gcloud.cmd compute ssh $Vm `
    --zone $Zone `
    --project $Project `
    --command $remoteCommand

  $exit = $LASTEXITCODE
  if ($exit -eq 0) {
    return
  }

  Write-Host "Bootstrap exited with code $exit. If this was the driver reboot, waiting for SSH and retrying once."
  Wait-ForSsh -Zone $Zone

  & gcloud.cmd compute ssh $Vm `
    --zone $Zone `
    --project $Project `
    --command $remoteCommand

  if ($LASTEXITCODE -ne 0) {
    throw "Bootstrap failed after retry."
  }
}

Invoke-Gcloud config set project $Project

$createdZone = Get-ExistingZone
if (-not $createdZone) {
  foreach ($zone in $Zones) {
    Write-Host "Trying zone $zone"
    & gcloud.cmd compute instances create $Vm `
      --project $Project `
      --zone $zone `
      --machine-type $Machine `
      --accelerator "type=nvidia-l4,count=1" `
      --maintenance-policy TERMINATE `
      --provisioning-model STANDARD `
      --boot-disk-size $BootDiskSize `
      --boot-disk-type $BootDiskType `
      --image-family ubuntu-2204-lts `
      --image-project ubuntu-os-cloud `
      --scopes cloud-platform `
      --tags $Tag

    if ($LASTEXITCODE -eq 0) {
      $createdZone = $zone
      break
    }
  }
}

if (-not $createdZone) {
  throw "Could not create VM in any configured zone."
}

Invoke-Gcloud config set compute/zone $createdZone
Invoke-Gcloud compute instances add-tags $Vm --zone $createdZone --project $Project --tags $Tag

$publicIp = (Invoke-RestMethod -Uri "https://ifconfig.me/ip").Trim()
Ensure-FirewallRule `
  -Name "allow-llm-craft-all-in-one-http" `
  -Allow "tcp:80" `
  -SourceRanges "0.0.0.0/0" `
  -Description "Public HTTP access for llm-craft web"

Ensure-FirewallRule `
  -Name "allow-llm-craft-all-in-one-ssh" `
  -Allow "tcp:22" `
  -SourceRanges "$publicIp/32" `
  -Description "Temporary SSH access from current public IP"

if (-not $SkipBootstrap) {
  Invoke-Bootstrap -Zone $createdZone
}

$ip = & gcloud.cmd compute instances describe $Vm `
  --zone $createdZone `
  --project $Project `
  --format "value(networkInterfaces[0].accessConfigs[0].natIP)"

Write-Host ""
Write-Host "VM ready"
Write-Host "Name: $Vm"
Write-Host "Zone: $createdZone"
Write-Host "IP: $ip"
Write-Host "Web: http://$ip/"
Write-Host ""
Write-Host "Model API is bound to 127.0.0.1:8000 inside the VM and is not exposed by firewall."
