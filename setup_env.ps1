# Setup environment variables for MMS Prime generation
Write-Host "Setting up environment variables..." -ForegroundColor Green

# Set paths
$env:MMS_INPUT_DIR = "C:/Users/sikan/Documents/Waleed/Thesis/Implimentation/MMS Player/MMS-examples"
$env:AVASAG_CORPUS_DIR = "C:/Users/sikan/Documents/Waleed/Thesis/Implimentation/AVASAGcorpus-MLsubset91/AVASAGcorpus-MLsubset91"
$env:MMS_OUTPUT_DIR = "C:/Users/sikan/Documents/Waleed/Thesis/Implimentation/MMS Player/inflected_mms"
$env:POINTING_DICT = "C:/Users/sikan/Documents/Waleed/Thesis/Implimentation/MMS Player/pointing_dict.json"
$env:BLENDER_EXE="C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
# Verify if paths exist
$paths = @{
    "MMS_INPUT_DIR" = $env:MMS_INPUT_DIR
    "AVASAG_CORPUS_DIR" = $env:AVASAG_CORPUS_DIR
    "MMS_OUTPUT_DIR" = $env:MMS_OUTPUT_DIR
    "POINTING_DICT" = $env:POINTING_DICT
    "BLENDER_EXE" = $env:BLENDER_EXE
}

# Check each path
foreach ($key in $paths.Keys) {
    $path = $paths[$key]
    if (Test-Path $path) {
        Write-Host "$key is set and exists: $path" -ForegroundColor Green
    } else {
        Write-Host "$key path does not exist: $path" -ForegroundColor Red
        Write-Host "Please update the path in setup_env.ps1" -ForegroundColor Yellow
    }
}

Write-Host "`nEnvironment setup complete!" -ForegroundColor Green
Write-Host "You can now run your Python script using:" -ForegroundColor Cyan
Write-Host "python generate_mms_prime.py --input-mms `"$env:MMS_INPUT_DIR\HandReloc-INDEX-X9.mms.csv`" --corpus-dir `"$env:AVASAG_CORPUS_DIR`" --output-dir `"$env:MMS_OUTPUT_DIR`" --pointing-dict `"$env:POINTING_DICT`"" -ForegroundColor Yellow 